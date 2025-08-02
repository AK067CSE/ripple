"""Configuration management for the Ripplica system."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # System Configuration
    default_llm_provider: str = Field(default="huggingface", env="DEFAULT_LLM_PROVIDER")
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    
    # Web Scraping
    max_concurrent_requests: int = 3
    request_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Vector Database
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()