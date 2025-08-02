"""Hybrid Ripplica engine that works with or without advanced dependencies."""

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
from .hybrid_vector_store import HybridVectorStore

# Try to import advanced components
try:
    from .web_scraper import WebScraper
    from .llm_providers import LLMManager
    ADVANCED_MODE = True
    print("✅ Advanced mode available (Web scraping + LLM)")
except ImportError as e:
    ADVANCED_MODE = False
    print(f"⚠️  Advanced components not available, using hybrid mode: {e}")


class HybridRipplicaEngine:
    """Hybrid Ripplica engine that adapts to available dependencies."""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.vector_store = HybridVectorStore()
        
        # Initialize advanced components if available
        self.llm_manager = None
        if ADVANCED_MODE:
            try:
                self.llm_manager = LLMManager()
                print("✅ LLM Manager initialized")
            except Exception as e:
                print(f"⚠️  LLM Manager failed to initialize: {e}")
        
        # Load existing vector store if available
        try:
            self.vector_store.load_from_disk("hybrid_ripplica_store.json")
        except Exception:
            pass  # Start with empty store
    
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
        
        if ADVANCED_MODE:
            pages = await self._advanced_web_search(cleaned_query, request.search_engine, request.max_results)
        else:
            pages = await self._simple_web_search(cleaned_query, request.max_results)
        
        search_time = time.time() - search_start
        
        # Generate response
        if ADVANCED_MODE and self.llm_manager:
            response_text, provider_used = await self._advanced_response_generation(
                cleaned_query, pages, request.llm_provider
            )
        else:
            response_text = self._simple_response_generation(cleaned_query, pages)
            provider_used = LLMProvider.HUGGINGFACE  # Default for simple mode
        
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
                llm_provider=provider_used
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
            llm_provider=provider_used,
            search_time=search_time,
            processing_time=processing_time,
            cached=False,
            similar_queries=similar_queries
        )
    
    async def _advanced_web_search(
        self, 
        query: str, 
        search_engine: SearchEngine, 
        max_results: int
    ) -> List[WebPage]:
        """Perform advanced web search using Playwright."""
        try:
            async with WebScraper() as scraper:
                search_result = await scraper.search_and_scrape(
                    query, search_engine, max_results
                )
                return search_result.pages
        except Exception as e:
            print(f"⚠️  Advanced web search failed, using simple search: {e}")
            return await self._simple_web_search(query, max_results)
    
    async def _simple_web_search(self, query: str, max_results: int = 5) -> List[WebPage]:
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
                            content=f"Search performed for query: {query}. Limited results available in hybrid mode.",
                            word_count=10
                        ))
        
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback response
            pages.append(WebPage(
                url='https://example.com',
                title=f"Search for: {query}",
                content=f"Unable to perform web search for '{query}'. This is a hybrid mode response.",
                word_count=15
            ))
        
        return pages[:max_results]
    
    async def _advanced_response_generation(
        self, 
        query: str, 
        pages: List[WebPage], 
        llm_provider: Optional[LLMProvider]
    ) -> Tuple[str, LLMProvider]:
        """Generate response using advanced LLM."""
        try:
            response_text, provider_used = await self.llm_manager.generate_response(
                query, pages, llm_provider
            )
            return response_text, provider_used
        except Exception as e:
            print(f"⚠️  Advanced response generation failed, using simple mode: {e}")
            return self._simple_response_generation(query, pages), LLMProvider.HUGGINGFACE
    
    def _simple_response_generation(self, query: str, pages: List[WebPage]) -> str:
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
    
    def _calculate_confidence_score(
        self, 
        pages: List[WebPage], 
        response: str, 
        search_time: float
    ) -> float:
        """Calculate confidence score for a response."""
        
        base_score = 0.4 if ADVANCED_MODE else 0.3  # Higher base for advanced mode
        
        # Factor 1: Number of sources
        if len(pages) >= 3:
            base_score += 0.2
        elif len(pages) >= 1:
            base_score += 0.1
        
        # Factor 2: Content quality
        if pages:
            avg_word_count = sum(page.word_count for page in pages) / len(pages)
            if avg_word_count > 100:
                base_score += 0.2
            elif avg_word_count > 50:
                base_score += 0.15
            elif avg_word_count > 20:
                base_score += 0.1
        
        # Factor 3: Response length
        response_words = len(response.split())
        if response_words > 100:
            base_score += 0.15
        elif response_words > 50:
            base_score += 0.1
        elif response_words > 20:
            base_score += 0.05
        
        # Factor 4: Search speed
        if search_time < 3:
            base_score += 0.05
        
        return min(max(base_score, 0.0), 1.0)
    
    def get_similar_queries(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get similar queries from the vector store."""
        return self.vector_store.find_similar_queries(query, k)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        vector_stats = self.vector_store.get_stats()
        
        available_providers = []
        if self.llm_manager:
            available_providers = [p.value for p in self.llm_manager.get_available_providers()]
        
        return {
            **vector_stats,
            'available_llm_providers': available_providers,
            'default_provider': settings.default_llm_provider,
            'similarity_threshold': settings.similarity_threshold,
            'advanced_mode': ADVANCED_MODE,
            'llm_available': self.llm_manager is not None
        }
    
    def save_state(self):
        """Save the current state to disk."""
        try:
            self.vector_store.save_to_disk("hybrid_ripplica_store.json")
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cached responses."""
        self.vector_store.cleanup_old_cache(max_age_days)
    
    async def batch_process_queries(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch."""
        responses = []
        
        for query in queries:
            try:
                request = QueryRequest(query=query)
                response = await self.process_query(request)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = QueryResponse(
                    query=query,
                    response=f"Error processing query: {str(e)}",
                    sources=[],
                    confidence_score=0.0,
                    llm_provider=LLMProvider.HUGGINGFACE,
                    search_time=0.0,
                    processing_time=0.0,
                    cached=False
                )
                responses.append(error_response)
        
        return responses