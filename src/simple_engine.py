"""Simplified Ripplica engine for basic functionality."""

import time
import asyncio
import aiohttp
from typing import List, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

from .models import (
    QueryRequest, QueryResponse, SearchEngine, LLMProvider, 
    CachedResponse, WebPage
)
from .config import settings
from .query_processor import QueryProcessor
from .simple_vector_store import SimpleVectorStore


class SimpleRipplicaEngine:
    """Simplified Ripplica engine with basic web scraping and caching."""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.vector_store = SimpleVectorStore()
        
        # Load existing vector store if available
        try:
            self.vector_store.load_from_disk("simple_ripplica_store.json")
        except Exception:
            pass  # Start with empty store
    
    async def simple_web_search(self, query: str, max_results: int = 5) -> List[WebPage]:
        """Perform a simple web search using DuckDuckGo."""
        
        pages = []
        
        try:
            # Use DuckDuckGo instant answer API
            async with aiohttp.ClientSession() as session:
                params = {
                    'q': query,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get('https://api.duckduckgo.com/', params=params) as response:
                    data = await response.json()
                    
                    # Extract abstract if available
                    if data.get('Abstract'):
                        pages.append(WebPage(
                            url=data.get('AbstractURL', 'https://duckduckgo.com'),
                            title=data.get('Heading', query),
                            content=data['Abstract'],
                            word_count=len(data['Abstract'].split())
                        ))
                    
                    # Extract from related topics
                    for topic in data.get('RelatedTopics', [])[:max_results-1]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            pages.append(WebPage(
                                url=topic.get('FirstURL', 'https://duckduckgo.com'),
                                title=topic.get('Text', '').split(' - ')[0][:100],
                                content=topic.get('Text', ''),
                                word_count=len(topic.get('Text', '').split())
                            ))
                    
                    # If we don't have enough results, create a simple response
                    if not pages:
                        pages.append(WebPage(
                            url='https://duckduckgo.com',
                            title=f"Search results for: {query}",
                            content=f"Search performed for query: {query}. Limited results available in simplified mode.",
                            word_count=10
                        ))
        
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback response
            pages.append(WebPage(
                url='https://example.com',
                title=f"Search for: {query}",
                content=f"Unable to perform web search for '{query}'. This is a simplified mode response.",
                word_count=15
            ))
        
        return pages[:max_results]
    
    def simple_response_generation(self, query: str, pages: List[WebPage]) -> str:
        """Generate a simple response based on scraped content."""
        
        if not pages:
            return f"I couldn't find specific information about '{query}', but this appears to be a valid question that would benefit from web research."
        
        # Extract key information from pages
        all_content = []
        sources = []
        
        for page in pages:
            if page.content and len(page.content.strip()) > 20:
                # Take first few sentences
                sentences = page.content.split('.')[:3]
                content_snippet = '. '.join(s.strip() for s in sentences if s.strip())
                if content_snippet:
                    all_content.append(content_snippet)
                    sources.append(page.title)
        
        if all_content:
            # Create a simple summary
            response = f"Based on the available information about '{query}':\n\n"
            
            for i, content in enumerate(all_content[:3], 1):
                response += f"{i}. {content}\n\n"
            
            if sources:
                response += f"Sources: {', '.join(sources[:3])}"
            
            return response
        else:
            return f"I found some information about '{query}', but the content was not detailed enough to provide a comprehensive answer. You may want to search for more specific terms or try a different query."
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query request and return a response."""
        
        start_time = time.time()
        
        # Validate and clean query
        is_valid, error_msg = self.query_processor.validate_query(request.query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")
        
        cleaned_query = self.query_processor.clean_query(request.query)
        query_intent = self.query_processor.classify_intent(cleaned_query)
        
        # Check for cached response first
        cached_response = None
        
        if request.use_cache:
            # Check exact match
            cached_response = self.vector_store.get_cached_response(cleaned_query)
            
            # Check similar queries if no exact match
            if not cached_response:
                similar_result = self.vector_store.find_cached_similar_response(cleaned_query)
                if similar_result:
                    similar_cached, similarity_score = similar_result
                    if similarity_score >= settings.similarity_threshold:
                        cached_response = similar_cached
        
        if cached_response:
            # Return cached response
            processing_time = time.time() - start_time
            
            # Get similar queries for context
            similar_queries = [q for q, _ in self.vector_store.find_similar_queries(cleaned_query, k=3)]
            
            return QueryResponse(
                query=request.query,
                response=cached_response.response,
                sources=cached_response.sources,
                confidence_score=cached_response.confidence_score,
                llm_provider=cached_response.llm_provider,
                search_time=0.0,
                processing_time=processing_time,
                cached=True,
                similar_queries=similar_queries
            )
        
        # Perform web search
        search_start = time.time()
        pages = await self.simple_web_search(cleaned_query, request.max_results)
        search_time = time.time() - search_start
        
        # Generate response
        response_text = self.simple_response_generation(cleaned_query, pages)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(pages, response_text, search_time)
        
        # Extract sources
        sources = [page.url for page in pages]
        
        # Cache the response
        if request.use_cache:
            cached_response = CachedResponse(
                query=cleaned_query,
                response=response_text,
                sources=sources,
                confidence_score=confidence_score,
                llm_provider=LLMProvider.HUGGINGFACE  # Default for simple mode
            )
            self.vector_store.cache_response(cached_response)
            
            # Add query embedding
            self.vector_store.add_query_embedding(cleaned_query, query_intent.value)
        
        # Get similar queries
        similar_queries = [q for q, _ in self.vector_store.find_similar_queries(cleaned_query, k=3)]
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources,
            confidence_score=confidence_score,
            llm_provider=LLMProvider.HUGGINGFACE,
            search_time=search_time,
            processing_time=processing_time,
            cached=False,
            similar_queries=similar_queries
        )
    
    def _calculate_confidence_score(
        self, 
        pages: List[WebPage], 
        response: str, 
        search_time: float
    ) -> float:
        """Calculate confidence score for a response."""
        
        score = 0.3  # Base score for simple mode
        
        # Factor 1: Number of sources
        if len(pages) >= 3:
            score += 0.2
        elif len(pages) >= 1:
            score += 0.1
        
        # Factor 2: Content quality
        if pages:
            avg_word_count = sum(page.word_count for page in pages) / len(pages)
            if avg_word_count > 50:
                score += 0.15
            elif avg_word_count > 20:
                score += 0.1
        
        # Factor 3: Response length
        response_words = len(response.split())
        if response_words > 50:
            score += 0.15
        elif response_words > 20:
            score += 0.1
        
        # Factor 4: Search speed
        if search_time < 3:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def get_similar_queries(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get similar queries from the vector store."""
        return self.vector_store.find_similar_queries(query, k)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            **vector_stats,
            'available_llm_providers': ['simple_mode'],
            'default_provider': 'simple_mode',
            'similarity_threshold': settings.similarity_threshold,
            'mode': 'simplified'
        }
    
    def save_state(self):
        """Save the current state to disk."""
        try:
            self.vector_store.save_to_disk("simple_ripplica_store.json")
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cached responses."""
        self.vector_store.cleanup_old_cache(max_age_days)