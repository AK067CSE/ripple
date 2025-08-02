"""Data models for the Ripplica system."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QueryIntent(str, Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    HOW_TO = "how_to"
    DEFINITION = "definition"
    NEWS = "news"
    OPINION = "opinion"
    OTHER = "other"


class SearchEngine(str, Enum):
    """Supported search engines."""
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class WebPage(BaseModel):
    """Represents a scraped web page."""
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    word_count: int = 0
    
    def __post_init__(self):
        if self.content:
            self.word_count = len(self.content.split())


class SearchResult(BaseModel):
    """Represents search engine results."""
    query: str
    pages: List[WebPage]
    search_engine: SearchEngine
    total_results: int
    search_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryEmbedding(BaseModel):
    """Query with its vector embedding."""
    query: str
    embedding: List[float]
    intent: Optional[QueryIntent] = None
    created_at: datetime = Field(default_factory=datetime.now)


class CachedResponse(BaseModel):
    """Cached query response."""
    query: str
    response: str
    sources: List[str]
    confidence_score: float
    llm_provider: LLMProvider
    created_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.now)


class QueryRequest(BaseModel):
    """API request model for queries."""
    query: str = Field(..., min_length=1, max_length=500)
    search_engine: SearchEngine = SearchEngine.GOOGLE
    llm_provider: Optional[LLMProvider] = None
    use_cache: bool = True
    max_results: int = Field(default=5, ge=1, le=10)


class QueryResponse(BaseModel):
    """API response model for queries."""
    query: str
    response: str
    sources: List[str]
    confidence_score: float
    llm_provider: LLMProvider
    search_time: float
    processing_time: float
    cached: bool = False
    similar_queries: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)