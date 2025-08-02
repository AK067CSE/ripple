"""Working vector store that gracefully handles ML dependencies."""

import pickle
import json
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

from .models import QueryEmbedding, CachedResponse
from .config import settings

# Global flag for ML availability
ML_AVAILABLE = False

# Try to import ML dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    ML_AVAILABLE = True
    print("✅ Advanced ML mode available")
except ImportError as e:
    print(f"⚠️  ML dependencies not available, using simple mode: {e}")


class WorkingVectorStore:
    """Vector store that works with or without ML dependencies."""
    
    def __init__(self):
        # Storage for query data
        self.query_embeddings: List[QueryEmbedding] = []
        self.cached_responses: Dict[str, CachedResponse] = {}
        
        # Simple similarity storage
        self.query_to_index: Dict[str, int] = {}
        self.index_to_query: Dict[int, str] = {}
        
        # ML components (if available)
        self.model = None
        self.faiss_index = None
        self.embedding_dim = 16  # Default for simple mode
        self.ml_mode = False
        
        if ML_AVAILABLE:
            try:
                self._init_ml_components()
            except Exception as e:
                print(f"⚠️  Failed to initialize ML components, using simple mode: {e}")
                self.ml_mode = False
    
    def _init_ml_components(self):
        """Initialize ML components if available."""
        try:
            # Use a lightweight model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.ml_mode = True
            
            print("✅ ML components initialized successfully")
        except Exception as e:
            print(f"⚠️  ML initialization failed: {e}")
            self.ml_mode = False
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using available method."""
        
        if self.ml_mode and self.model is not None:
            try:
                # Use sentence transformers
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️  ML embedding failed, using simple mode: {e}")
        
        # Fallback to simple hash-based embedding
        return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple hash-based embedding for text."""
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
        if len(embedding) < self.embedding_dim:
            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
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
        
        # Add to FAISS index if available
        if self.ml_mode and self.faiss_index is not None:
            try:
                embedding_array = np.array([embedding], dtype=np.float32)
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding_array)
                self.faiss_index.add(embedding_array)
            except Exception as e:
                print(f"⚠️  FAISS indexing failed: {e}")
        
        return index
    
    def find_similar_queries(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """Find similar queries using available method."""
        
        if threshold is None:
            threshold = settings.similarity_threshold
        
        if not self.query_embeddings:
            return []
        
        if self.ml_mode and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                return self._find_similar_ml(query, k, threshold)
            except Exception as e:
                print(f"⚠️  ML similarity search failed, using simple mode: {e}")
        
        # Fallback to simple similarity
        return self._find_similar_simple(query, k, threshold)
    
    def _find_similar_ml(self, query: str, k: int, threshold: float) -> List[Tuple[str, float]]:
        """Find similar queries using ML methods."""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_array, min(k + 1, self.faiss_index.ntotal))
        
        # Filter results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.query_embeddings):
                stored_query = self.query_embeddings[idx].query
                if stored_query != query and score >= threshold:
                    results.append((stored_query, float(score)))
        
        return results[:k]
    
    def _find_similar_simple(self, query: str, k: int, threshold: float) -> List[Tuple[str, float]]:
        """Find similar queries using simple text similarity."""
        # Calculate similarities
        similarities = []
        for stored_query in self.query_embeddings:
            if stored_query.query != query:  # Exclude exact match
                similarity = self._simple_similarity(query, stored_query.query)
                if similarity >= threshold:
                    similarities.append((stored_query.query, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
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
            'vector_dimension': self.embedding_dim,
            'ml_mode': self.ml_mode,
            'faiss_entries': self.faiss_index.ntotal if self.faiss_index else 0,
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
            'index_to_query': self.index_to_query,
            'ml_mode': self.ml_mode,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Save FAISS index separately if available
        if self.ml_mode and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            try:
                faiss_filepath = filepath.replace('.json', '.faiss')
                faiss.write_index(self.faiss_index, faiss_filepath)
            except Exception as e:
                print(f"⚠️  Failed to save FAISS index: {e}")
    
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
            
            # Restore FAISS index if available
            if self.ml_mode and self.faiss_index is not None:
                try:
                    faiss_filepath = filepath.replace('.json', '.faiss')
                    import os
                    if os.path.exists(faiss_filepath):
                        self.faiss_index = faiss.read_index(faiss_filepath)
                        print("✅ FAISS index loaded successfully")
                except Exception as e:
                    print(f"⚠️  Failed to load FAISS index: {e}")
            
            print(f"✅ Vector store loaded: {len(self.query_embeddings)} queries, {len(self.cached_responses)} cached responses")
            
        except FileNotFoundError:
            print(f"Vector store file not found: {filepath}")
        except Exception as e:
            print(f"Error loading vector store: {e}")