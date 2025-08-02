"""Simplified vector storage without heavy ML dependencies."""

import pickle
import json
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib

from .models import QueryEmbedding, CachedResponse
from .config import settings


class SimpleVectorStore:
    """Simplified vector store using basic string similarity."""
    
    def __init__(self):
        # Storage for query data
        self.query_embeddings: List[QueryEmbedding] = []
        self.cached_responses: Dict[str, CachedResponse] = {}
        
        # Simple similarity storage
        self.query_to_index: Dict[str, int] = {}
        self.index_to_query: Dict[int, str] = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a simple hash-based embedding for text.
        This is a placeholder until proper embeddings are available.
        """
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.lower().encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to float values (normalized)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            hex_pair = hash_hex[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to fixed size
        target_size = 16  # Simple embedding size
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding
    
    def add_query_embedding(self, query: str, intent: Optional[str] = None) -> int:
        """Add a query embedding to the vector store."""
        # Check if query already exists
        if query in self.query_to_index:
            return self.query_to_index[query]
        
        # Generate embedding
        embedding = self.generate_embedding(query)
        
        # Create QueryEmbedding object
        query_embedding = QueryEmbedding(
            query=query,
            embedding=embedding,
            intent=intent
        )
        
        # Add to storage
        index = len(self.query_embeddings)
        self.query_embeddings.append(query_embedding)
        
        # Update mappings
        self.query_to_index[query] = index
        self.index_to_query[index] = query
        
        return index
    
    def simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_queries(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Find similar queries using simple text similarity."""
        if threshold is None:
            threshold = settings.similarity_threshold
        
        if not self.query_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for stored_query in self.query_embeddings:
            if stored_query.query != query:  # Exclude exact match
                similarity = self.simple_similarity(query, stored_query.query)
                if similarity >= threshold:
                    similarities.append((stored_query.query, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def cache_response(self, response: CachedResponse):
        """Cache a query response."""
        self.cached_responses[response.query] = response
        
        # Add query embedding if not exists
        if response.query not in self.query_to_index:
            self.add_query_embedding(response.query)
    
    def get_cached_response(self, query: str) -> Optional[CachedResponse]:
        """Get cached response for a query."""
        if query in self.cached_responses:
            cached = self.cached_responses[query]
            # Update access statistics
            cached.access_count += 1
            cached.last_accessed = datetime.now()
            return cached
        
        return None
    
    def find_cached_similar_response(
        self, 
        query: str, 
        threshold: float = None
    ) -> Optional[Tuple[CachedResponse, float]]:
        """Find cached response for similar query."""
        if threshold is None:
            threshold = settings.similarity_threshold
        
        similar_queries = self.find_similar_queries(query, k=1, threshold=threshold)
        
        if similar_queries:
            similar_query, similarity = similar_queries[0]
            if similar_query in self.cached_responses:
                cached_response = self.cached_responses[similar_query]
                return cached_response, similarity
        
        return None
    
    def cleanup_old_cache(self, max_age_days: int = 7):
        """Remove old cached responses."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Find expired entries
        expired_queries = []
        for query, response in self.cached_responses.items():
            if response.created_at < cutoff_date:
                expired_queries.append(query)
        
        # Remove expired entries
        for query in expired_queries:
            del self.cached_responses[query]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_queries': len(self.query_embeddings),
            'cached_responses': len(self.cached_responses),
            'vector_dimension': 16,  # Simple embedding size
            'index_size': len(self.query_embeddings),
            'most_accessed_queries': sorted(
                [(q, r.access_count) for q, r in self.cached_responses.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def save_to_disk(self, filepath: str):
        """Save vector store to disk."""
        data = {
            'query_embeddings': [
                {
                    'query': qe.query,
                    'embedding': qe.embedding,
                    'intent': qe.intent,
                    'created_at': qe.created_at.isoformat()
                } for qe in self.query_embeddings
            ],
            'cached_responses': {
                query: {
                    'query': resp.query,
                    'response': resp.response,
                    'sources': resp.sources,
                    'confidence_score': resp.confidence_score,
                    'llm_provider': resp.llm_provider.value,
                    'created_at': resp.created_at.isoformat(),
                    'access_count': resp.access_count,
                    'last_accessed': resp.last_accessed.isoformat()
                } for query, resp in self.cached_responses.items()
            },
            'query_to_index': self.query_to_index,
            'index_to_query': self.index_to_query
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_from_disk(self, filepath: str):
        """Load vector store from disk."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct query embeddings
            self.query_embeddings = []
            for qe_data in data.get('query_embeddings', []):
                qe = QueryEmbedding(
                    query=qe_data['query'],
                    embedding=qe_data['embedding'],
                    intent=qe_data.get('intent'),
                    created_at=datetime.fromisoformat(qe_data['created_at'])
                )
                self.query_embeddings.append(qe)
            
            # Reconstruct cached responses
            from .models import LLMProvider
            self.cached_responses = {}
            for query, resp_data in data.get('cached_responses', {}).items():
                resp = CachedResponse(
                    query=resp_data['query'],
                    response=resp_data['response'],
                    sources=resp_data['sources'],
                    confidence_score=resp_data['confidence_score'],
                    llm_provider=LLMProvider(resp_data['llm_provider']),
                    created_at=datetime.fromisoformat(resp_data['created_at']),
                    access_count=resp_data['access_count'],
                    last_accessed=datetime.fromisoformat(resp_data['last_accessed'])
                )
                self.cached_responses[query] = resp
            
            self.query_to_index = data.get('query_to_index', {})
            self.index_to_query = {int(k): v for k, v in data.get('index_to_query', {}).items()}
            
        except FileNotFoundError:
            print(f"Vector store file not found: {filepath}")
        except Exception as e:
            print(f"Error loading vector store: {e}")