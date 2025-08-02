"""Main Ripplica engine that orchestrates all components."""

import time
from typing import List, Optional, Tuple
from datetime import datetime

from .models import (
    QueryRequest, QueryResponse, SearchEngine, LLMProvider, 
    CachedResponse, WebPage
)
from .config import settings
from .query_processor import QueryProcessor
from .web_scraper import WebScraper
from .vector_store import VectorStore
from .llm_providers import LLMManager


class RipplicaEngine:
    """Main engine that orchestrates query processing, web scraping, and response generation."""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        
        # Load existing vector store if available
        try:
            self.vector_store.load_from_disk("ripplica_store.pkl")
        except Exception:
            pass  # Start with empty store
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query request and return a response.
        
        Args:
            request: QueryRequest object
            
        Returns:
            QueryResponse object
        """
        start_time = time.time()
        
        # Validate and clean query
        is_valid, error_msg = self.query_processor.validate_query(request.query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")
        
        cleaned_query = self.query_processor.clean_query(request.query)
        query_intent = self.query_processor.classify_intent(cleaned_query)
        
        # Check for cached response first
        cached_response = None
        similar_cached = None
        
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
        
        # Perform web search and scraping
        search_start = time.time()
        
        async with WebScraper() as scraper:
            search_result = await scraper.search_and_scrape(
                cleaned_query,
                request.search_engine,
                request.max_results
            )
        
        search_time = time.time() - search_start
        
        if not search_result.pages:
            raise ValueError("No web pages could be scraped for this query")
        
        # Generate response using LLM
        llm_provider = request.llm_provider or LLMProvider(settings.default_llm_provider)
        
        try:
            response_text, provider_used = await self.llm_manager.generate_response(
                cleaned_query,
                search_result.pages,
                llm_provider
            )
        except Exception as e:
            # Fallback to best available provider
            try:
                response_text, provider_used = await self.llm_manager.generate_response(
                    cleaned_query,
                    search_result.pages
                )
            except Exception as fallback_error:
                raise ValueError(f"Failed to generate response: {str(fallback_error)}")
        
        # Calculate confidence score based on various factors
        confidence_score = self._calculate_confidence_score(
            search_result.pages,
            response_text,
            search_result.search_time
        )
        
        # Extract sources
        sources = [page.url for page in search_result.pages]
        
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
    
    def _calculate_confidence_score(
        self, 
        pages: List[WebPage], 
        response: str, 
        search_time: float
    ) -> float:
        """
        Calculate confidence score for a response.
        
        Args:
            pages: List of scraped web pages
            response: Generated response text
            search_time: Time taken for web search
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Factor 1: Number of sources
        if len(pages) >= 3:
            score += 0.2
        elif len(pages) >= 2:
            score += 0.1
        
        # Factor 2: Content quality (word count)
        avg_word_count = sum(page.word_count for page in pages) / len(pages) if pages else 0
        if avg_word_count > 500:
            score += 0.15
        elif avg_word_count > 200:
            score += 0.1
        
        # Factor 3: Response length (indicates comprehensive answer)
        response_words = len(response.split())
        if response_words > 100:
            score += 0.1
        elif response_words > 50:
            score += 0.05
        
        # Factor 4: Search speed (faster = more reliable sources)
        if search_time < 5:
            score += 0.05
        
        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)
    
    def get_similar_queries(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get similar queries from the vector store."""
        return self.vector_store.find_similar_queries(query, k)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        vector_stats = self.vector_store.get_stats()
        available_providers = self.llm_manager.get_available_providers()
        
        return {
            **vector_stats,
            'available_llm_providers': [p.value for p in available_providers],
            'default_provider': settings.default_llm_provider,
            'similarity_threshold': settings.similarity_threshold
        }
    
    def save_state(self):
        """Save the current state to disk."""
        try:
            self.vector_store.save_to_disk("ripplica_store.pkl")
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cached responses."""
        self.vector_store.cleanup_old_cache(max_age_days)
    
    async def batch_process_queries(self, queries: List[str]) -> List[QueryResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of QueryResponse objects
        """
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