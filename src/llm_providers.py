"""LLM providers for response generation."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import openai
import google.generativeai as genai
from groq import Groq
import requests
from transformers import pipeline

from .models import LLMProvider, WebPage
from .config import settings


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int = 500
    ) -> str:
        """Generate response based on query and context."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self):
        self.client = None
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    async def generate_response(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int = 500
    ) -> str:
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        # Prepare context
        context_text = self._prepare_context(context)
        
        prompt = f"""Based on the following web search results, provide a comprehensive answer to the query.

Query: {query}

Search Results:
{context_text}

Please provide a well-structured answer that:
1. Directly addresses the query
2. Uses information from the search results
3. Is factual and informative
4. Cites relevant sources when possible

Answer:"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _prepare_context(self, context: List[WebPage]) -> str:
        """Prepare context from web pages."""
        context_parts = []
        for i, page in enumerate(context[:5], 1):
            content = page.content[:1000] + "..." if len(page.content) > 1000 else page.content
            context_parts.append(f"Source {i} ({page.url}):\nTitle: {page.title}\nContent: {content}\n")
        
        return "\n".join(context_parts)


class GroqProvider(BaseLLMProvider):
    """Groq provider for fast inference."""
    
    def __init__(self):
        self.client = None
        if settings.groq_api_key:
            self.client = Groq(api_key=settings.groq_api_key)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    async def generate_response(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int = 500
    ) -> str:
        if not self.is_available():
            raise ValueError("Groq API key not configured")
        
        context_text = self._prepare_context(context)
        
        prompt = f"""Answer the following query based on the provided search results.

Query: {query}

Search Results:
{context_text}

Provide a clear, concise answer that addresses the query directly:"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _prepare_context(self, context: List[WebPage]) -> str:
        """Prepare context from web pages."""
        context_parts = []
        for i, page in enumerate(context[:5], 1):
            content = page.content[:800] + "..." if len(page.content) > 800 else page.content
            context_parts.append(f"{i}. {page.title} ({page.url})\n{content}\n")
        
        return "\n".join(context_parts)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider."""
    
    def __init__(self):
        self.model = None
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
    
    def is_available(self) -> bool:
        return self.model is not None
    
    async def generate_response(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int = 500
    ) -> str:
        if not self.is_available():
            raise ValueError("Google API key not configured")
        
        context_text = self._prepare_context(context)
        
        prompt = f"""Based on the web search results below, answer the query comprehensively.

Query: {query}

Web Search Results:
{context_text}

Please provide a detailed answer that synthesizes information from the sources:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return response.text.strip()
            
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")
    
    def _prepare_context(self, context: List[WebPage]) -> str:
        """Prepare context from web pages."""
        context_parts = []
        for i, page in enumerate(context[:5], 1):
            content = page.content[:1000] + "..." if len(page.content) > 1000 else page.content
            context_parts.append(f"Source {i}: {page.title}\nURL: {page.url}\nContent: {content}\n")
        
        return "\n".join(context_parts)


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider using free models."""
    
    def __init__(self):
        self.api_key = settings.huggingface_api_key
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.local_pipeline = None
        
        # Try to initialize local pipeline as fallback
        try:
            self.local_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer="microsoft/DialoGPT-small"
            )
        except Exception:
            pass
    
    def is_available(self) -> bool:
        return self.api_key is not None or self.local_pipeline is not None
    
    async def generate_response(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int = 500
    ) -> str:
        # Try API first, then local model
        if self.api_key:
            try:
                return await self._generate_with_api(query, context, max_tokens)
            except Exception as e:
                print(f"HuggingFace API failed: {e}")
        
        if self.local_pipeline:
            return await self._generate_with_local(query, context, max_tokens)
        
        raise ValueError("No HuggingFace provider available")
    
    async def _generate_with_api(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int
    ) -> str:
        """Generate response using HuggingFace API."""
        context_text = self._prepare_context(context)
        
        prompt = f"Query: {query}\n\nContext: {context_text}\n\nAnswer:"
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        async with asyncio.to_thread(requests.post, self.api_url, headers=headers, json=payload) as response:
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
            
            raise Exception(f"HuggingFace API error: {response.status_code}")
    
    async def _generate_with_local(
        self, 
        query: str, 
        context: List[WebPage], 
        max_tokens: int
    ) -> str:
        """Generate response using local model."""
        context_text = self._prepare_context(context)
        
        # Simple summarization approach
        summary_parts = []
        for page in context[:3]:
            # Extract key sentences
            sentences = page.content.split('.')[:3]
            summary_parts.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        # Create a simple response
        if summary_parts:
            response = f"Based on the search results: {' '.join(summary_parts[:5])}"
            return response[:max_tokens]
        
        return "I found some relevant information but couldn't generate a comprehensive response."
    
    def _prepare_context(self, context: List[WebPage]) -> str:
        """Prepare context from web pages."""
        context_parts = []
        for page in context[:3]:
            content = page.content[:500] + "..." if len(page.content) > 500 else page.content
            context_parts.append(f"{page.title}: {content}")
        
        return " | ".join(context_parts)


class LLMManager:
    """Manages multiple LLM providers."""
    
    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: OpenAIProvider(),
            LLMProvider.GROQ: GroqProvider(),
            LLMProvider.GOOGLE: GoogleProvider(),
            LLMProvider.HUGGINGFACE: HuggingFaceProvider()
        }
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return [
            provider for provider, instance in self.providers.items()
            if instance.is_available()
        ]
    
    def get_provider(self, provider_name: LLMProvider) -> BaseLLMProvider:
        """Get a specific provider instance."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider = self.providers[provider_name]
        if not provider.is_available():
            raise ValueError(f"Provider {provider_name} is not available")
        
        return provider
    
    def get_best_available_provider(self) -> BaseLLMProvider:
        """Get the best available provider based on preference order."""
        preference_order = [
            LLMProvider.GROQ,      # Fast and good quality
            LLMProvider.OPENAI,    # High quality
            LLMProvider.GOOGLE,    # Good alternative
            LLMProvider.HUGGINGFACE # Free fallback
        ]
        
        for provider_name in preference_order:
            if provider_name in self.providers and self.providers[provider_name].is_available():
                return self.providers[provider_name]
        
        raise ValueError("No LLM providers available")
    
    async def generate_response(
        self,
        query: str,
        context: List[WebPage],
        provider: Optional[LLMProvider] = None,
        max_tokens: int = 500
    ) -> tuple[str, LLMProvider]:
        """
        Generate response using specified or best available provider.
        
        Returns:
            Tuple of (response, provider_used)
        """
        if provider:
            llm_provider = self.get_provider(provider)
            provider_used = provider
        else:
            llm_provider = self.get_best_available_provider()
            # Find which provider was used
            provider_used = None
            for name, instance in self.providers.items():
                if instance is llm_provider:
                    provider_used = name
                    break
        
        response = await llm_provider.generate_response(query, context, max_tokens)
        return response, provider_used