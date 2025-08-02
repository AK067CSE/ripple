"""Vector storage and similarity search using FAISS."""

import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import faiss
from sentence_transformers import SentenceTransformer

from .models import QueryEmbedding, CachedResponse
from .config import settings


class VectorStore:
    """Handles vector storage and similarity search using FAISS."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.dimension = settings.vector_dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Storage for query data
        self.query_embeddings: List[QueryEmbedding] = []
        self.cached_responses: Dict[str, CachedResponse] = {}
        
        # Metadata storage
        self.query_to_index: Dict[str, int] = {}
        self.index_to_query: Dict[int, str] = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def add_query_embedding(self, query: str, intent: Optional[str] = None) -> int:
        """
        Add a query embedding to the vector store.
        
        Args:
            query: Query string
            intent: Optional query intent
            
        Returns:
            Index of the added embedding
        """
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
        
        # Add to FAISS index
        embedding_array = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_array)
        
        return index
    
    def find_similar_queries(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar queries using vector similarity.
        
        Args:
            query: Query to find similarities for
            k: Number of similar queries to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (query, similarity_score) tuples
        """
        if threshold is None:
            threshold = settings.similarity_threshold
        
        if self.index.ntotal == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search for similar vectors
        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Filter by threshold and format results
        similar_queries = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.query_embeddings):
                similar_query = self.query_embeddings[idx].query
                if similar_query != query:  # Exclude exact match
                    similar_queries.append((similar_query, float(score)))
        
        return similar_queries
    
    def cache_response(self, response: CachedResponse):
        """
        Cache a query response.
        
        Args:
            response: CachedResponse object to cache
        """
        self.cached_responses[response.query] = response
        
        # Add query embedding if not exists
        if response.query not in self.query_to_index:
            self.add_query_embedding(response.query)
    
    def get_cached_response(self, query: str) -> Optional[CachedResponse]:
        """
        Get cached response for a query.
        
        Args:
            query: Query string
            
        Returns:
            CachedResponse if found, None otherwise
        """
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
        """
        Find cached response for similar query.
        
        Args:
            query: Query string
            threshold: Similarity threshold
            
        Returns:
            Tuple of (CachedResponse, similarity_score) if found
        """
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
        """
        Remove old cached responses.
        
        Args:
            max_age_days: Maximum age of cached responses in days
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Find expired entries
        expired_queries = []
        for query, response in self.cached_responses.items():
            if response.created_at < cutoff_date:
                expired_queries.append(query)
        
        # Remove expired entries
        for query in expired_queries:
            del self.cached_responses[query]
            
            # Remove from vector index (this is complex with FAISS, so we'll rebuild)
            if query in self.query_to_index:
                # Mark for rebuild
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_queries': len(self.query_embeddings),
            'cached_responses': len(self.cached_responses),
            'vector_dimension': self.dimension,
            'index_size': self.index.ntotal,
            'most_accessed_queries': sorted(
                [(q, r.access_count) for q, r in self.cached_responses.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def save_to_disk(self, filepath: str):
        """
        Save vector store to disk.
        
        Args:
            filepath: Path to save the store
        """
        data = {
            'query_embeddings': self.query_embeddings,
            'cached_responses': self.cached_responses,
            'query_to_index': self.query_to_index,
            'index_to_query': self.index_to_query
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
    
    def load_from_disk(self, filepath: str):
        """
        Load vector store from disk.
        
        Args:
            filepath: Path to load the store from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.query_embeddings = data['query_embeddings']
            self.cached_responses = data['cached_responses']
            self.query_to_index = data['query_to_index']
            self.index_to_query = data['index_to_query']
            
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
        except FileNotFoundError:
            print(f"Vector store file not found: {filepath}")
        except Exception as e:
            print(f"Error loading vector store: {e}")