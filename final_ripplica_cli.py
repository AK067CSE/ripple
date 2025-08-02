"""
🚀 FINAL RIPPLICA CLI - Complete AI Web Query System
====================================================

This is the complete, production-ready version of Ripplica that includes:
✅ ML-powered vector embeddings with FAISS
✅ Advanced web scraping with Playwright + fallback
✅ Multiple LLM providers (OpenAI, Groq, Google, Hugging Face)
✅ Intelligent caching and similarity matching
✅ Graceful degradation when APIs are not available
✅ Comprehensive error handling and logging

Usage:
    python final_ripplica_cli.py "your query here"    # Single query
    python final_ripplica_cli.py                      # Interactive mode

Environment Variables (optional):
    OPENAI_API_KEY=your_openai_key
    GROQ_API_KEY=your_groq_key  
    GOOGLE_API_KEY=your_google_key
    HUGGINGFACE_API_KEY=your_hf_key
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import QueryRequest, SearchEngine, LLMProvider
from src.query_processor import QueryProcessor
from src.working_vector_store import WorkingVectorStore
from src.working_web_scraper import WorkingWebScraper

# Try to import LLM providers
try:
    from src.llm_providers import LLMManager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class FinalRipplicaEngine:
    """Complete Ripplica engine with all features."""
    
    def __init__(self):
        print("🚀 Initializing Ripplica Engine...")
        
        self.query_processor = QueryProcessor()
        self.vector_store = WorkingVectorStore()
        
        # Initialize LLM manager if available
        self.llm_manager = None
        if LLM_AVAILABLE:
            try:
                self.llm_manager = LLMManager()
                available_providers = self.llm_manager.get_available_providers()
                if available_providers:
                    print(f"✅ LLM Manager initialized with: {[p.value for p in available_providers]}")
                else:
                    print("⚠️  LLM Manager initialized but no API keys found")
            except Exception as e:
                print(f"⚠️  LLM Manager failed: {e}")
        
        # Load existing data
        try:
            self.vector_store.load_from_disk("final_ripplica_store.json")
        except Exception:
            pass
        
        print("✅ Ripplica Engine ready!")
    
    async def process_query(self, request: QueryRequest):
        """Process a query with all available features."""
        
        import time
        start_time = time.time()
        
        # Validate and clean query
        is_valid, error_msg = self.query_processor.validate_query(request.query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")
        
        cleaned_query = self.query_processor.clean_query(request.query)
        query_intent = self.query_processor.classify_intent(cleaned_query)
        
        # Check cache first
        cached_response = None
        if request.use_cache:
            cached_response = self.vector_store.get_cached_response(cleaned_query)
            
            if not cached_response:
                similar_result = self.vector_store.find_cached_similar_response(cleaned_query)
                if similar_result:
                    similar_cached, similarity_score = similar_result
                    if similarity_score >= 0.7:  # High similarity threshold
                        cached_response = similar_cached
        
        if cached_response:
            processing_time = time.time() - start_time
            similar_queries = [q for q, _ in self.vector_store.find_similar_queries(cleaned_query, k=3)]
            
            return {
                'query': request.query,
                'response': cached_response.response,
                'sources': cached_response.sources,
                'confidence_score': cached_response.confidence_score,
                'llm_provider': cached_response.llm_provider.value,
                'search_time': 0.0,
                'processing_time': processing_time,
                'cached': True,
                'similar_queries': similar_queries
            }
        
        # Perform web search and scraping
        search_start = time.time()
        
        async with WorkingWebScraper() as scraper:
            search_result = await scraper.search_and_scrape(
                cleaned_query, 
                request.search_engine, 
                request.max_results
            )
        
        search_time = time.time() - search_start
        
        # Generate response
        provider_used = LLMProvider.HUGGINGFACE  # Default fallback
        
        if self.llm_manager and search_result.pages:
            try:
                response_text, provider_used = await self.llm_manager.generate_response(
                    cleaned_query, search_result.pages, request.llm_provider
                )
                print(f"✅ Response generated using {provider_used.value}")
            except Exception as e:
                print(f"⚠️  LLM generation failed ({e}), using fallback")
                response_text = self._generate_simple_response(cleaned_query, search_result.pages)
        else:
            response_text = self._generate_simple_response(cleaned_query, search_result.pages)
            print("✅ Response generated using simple fallback")
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(search_result.pages, response_text, search_time)
        
        # Cache the response
        if request.use_cache:
            from src.models import CachedResponse
            cached_response = CachedResponse(
                query=cleaned_query,
                response=response_text,
                sources=[page.url for page in search_result.pages],
                confidence_score=confidence_score,
                llm_provider=provider_used
            )
            self.vector_store.cache_response(cached_response)
            self.vector_store.add_query_embedding(cleaned_query, query_intent.value)
        
        # Get similar queries
        similar_queries = [q for q, _ in self.vector_store.find_similar_queries(cleaned_query, k=3)]
        
        processing_time = time.time() - start_time
        
        return {
            'query': request.query,
            'response': response_text,
            'sources': [page.url for page in search_result.pages],
            'confidence_score': confidence_score,
            'llm_provider': provider_used.value,
            'search_time': search_time,
            'processing_time': processing_time,
            'cached': False,
            'similar_queries': similar_queries
        }
    
    def _generate_simple_response(self, query: str, pages):
        """Generate a simple response from scraped content."""
        
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
            return f"I found some information about '{query}', but the content was not detailed enough to provide a comprehensive answer."
    
    def _calculate_confidence(self, pages, response: str, search_time: float) -> float:
        """Calculate confidence score."""
        
        base_score = 0.5
        
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
                base_score += 0.1
        
        # Factor 3: Response length
        response_words = len(response.split())
        if response_words > 100:
            base_score += 0.15
        elif response_words > 50:
            base_score += 0.1
        
        # Factor 4: Search speed
        if search_time < 5:
            base_score += 0.05
        
        return min(max(base_score, 0.0), 1.0)
    
    def get_stats(self):
        """Get system statistics."""
        stats = self.vector_store.get_stats()
        
        available_providers = []
        if self.llm_manager:
            available_providers = [p.value for p in self.llm_manager.get_available_providers()]
        
        return {
            **stats,
            'available_llm_providers': available_providers,
            'llm_available': self.llm_manager is not None,
            'web_scraping': True,
            'advanced_mode': True
        }
    
    def save_state(self):
        """Save state to disk."""
        try:
            self.vector_store.save_to_disk("final_ripplica_store.json")
        except Exception as e:
            print(f"Error saving state: {e}")


class FinalRipplicaCLI:
    """Final production-ready CLI."""
    
    def __init__(self):
        self.engine = FinalRipplicaEngine()
        self._print_banner()
    
    def _print_banner(self):
        """Print startup banner."""
        print("\n" + "="*60)
        print("🚀 RIPPLICA - AI Web Query System")
        print("="*60)
        print("Complete AI-powered web research and query system")
        print("Automatically adapts to available dependencies and APIs")
        print("="*60)
    
    async def run_query(self, query: str, use_cache: bool = True, max_results: int = 3):
        """Run a single query."""
        
        try:
            request = QueryRequest(
                query=query,
                search_engine=SearchEngine.DUCKDUCKGO,
                use_cache=use_cache,
                max_results=max_results
            )
            
            print(f"\n🔍 Processing query: {query}")
            print("=" * 60)
            
            response = await self.engine.process_query(request)
            
            self._print_response(response)
            
            # Save state
            self.engine.save_state()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def _print_response(self, response):
        """Print response in a beautiful format."""
        
        # Status indicators
        cache_status = "✅ CACHED" if response['cached'] else "🔄 FRESH"
        
        if response['confidence_score'] > 0.8:
            confidence_color = "🟢 HIGH"
        elif response['confidence_score'] > 0.6:
            confidence_color = "🟡 MEDIUM"
        else:
            confidence_color = "🔴 LOW"
        
        # Get system stats to show mode
        stats = self.engine.get_stats()
        mode_info = []
        if stats.get('ml_mode'):
            mode_info.append("ML")
        if stats.get('advanced_mode'):
            mode_info.append("Advanced")
        if stats.get('llm_available'):
            mode_info.append("LLM")
        
        mode_str = "+".join(mode_info) if mode_info else "Simple"
        
        # Print header
        print(f"📊 STATUS: {cache_status}")
        print(f"🎯 CONFIDENCE: {confidence_color} ({response['confidence_score']:.2f})")
        print(f"🤖 PROVIDER: {response['llm_provider'].upper()}")
        print(f"⚙️  MODE: {mode_str}")
        print(f"⏱️  SEARCH TIME: {response['search_time']:.2f}s")
        print(f"⚡ PROCESSING TIME: {response['processing_time']:.2f}s")
        print("\n" + "="*60)
        
        # Response
        print("🤖 AI RESPONSE:")
        print("-" * 60)
        print(response['response'])
        print()
        
        # Sources
        if response['sources']:
            print("📚 SOURCES:")
            print("-" * 60)
            for i, source in enumerate(response['sources'], 1):
                # Clean up source URLs for display
                display_url = source
                if len(display_url) > 80:
                    display_url = display_url[:77] + "..."
                print(f"{i}. {display_url}")
            print()
        
        # Similar queries
        if response['similar_queries']:
            print("🔗 SIMILAR QUERIES:")
            print("-" * 60)
            for query in response['similar_queries']:
                print(f"• {query}")
            print()
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        
        print("\n🎮 INTERACTIVE MODE")
        print("Type your queries below. Use 'help' for commands, 'quit' to exit")
        print("="*60)
        
        # Show current capabilities
        self._show_capabilities()
        
        while True:
            try:
                query = input("\n🔍 Query > ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Ripplica! Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._print_help()
                    continue
                
                if query.lower() == 'stats':
                    self._print_stats()
                    continue
                
                if query.lower() == 'capabilities':
                    self._show_capabilities()
                    continue
                
                if query.lower().startswith('clear'):
                    self.engine.vector_store.cleanup_old_cache(0)
                    print("✅ Cache cleared!")
                    continue
                
                # Process query
                await self.run_query(query)
                
            except KeyboardInterrupt:
                print("\n👋 Thank you for using Ripplica! Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _show_capabilities(self):
        """Show current system capabilities."""
        
        stats = self.engine.get_stats()
        
        print("\n🔧 CURRENT CAPABILITIES:")
        print("-" * 30)
        
        # Core features
        if stats.get('ml_mode'):
            print("✅ ML Embeddings (sentence-transformers)")
        else:
            print("⚪ Simple Text Embeddings")
        
        if stats.get('faiss_entries', 0) > 0:
            print(f"✅ FAISS Vector Search ({stats['faiss_entries']} entries)")
        else:
            print("⚪ Basic Vector Search")
        
        print("✅ Advanced Web Scraping (Playwright + fallback)")
        print("✅ Intelligent Response Caching")
        
        # LLM providers
        if stats.get('available_llm_providers'):
            print(f"✅ LLM Providers: {', '.join(stats['available_llm_providers'])}")
        else:
            print("⚪ Simple Response Generation (no API keys)")
        
        # Environment status
        print(f"\n📊 CACHE STATUS:")
        print(f"   • Total Queries: {stats['total_queries']}")
        print(f"   • Cached Responses: {stats['cached_responses']}")
        print(f"   • Vector Dimension: {stats['vector_dimension']}")
    
    def _print_help(self):
        """Print comprehensive help."""
        
        print("\n📖 RIPPLICA HELP")
        print("="*40)
        print("COMMANDS:")
        print("  help         - Show this help")
        print("  stats        - Show detailed statistics")
        print("  capabilities - Show current system capabilities")
        print("  clear        - Clear response cache")
        print("  quit/exit    - Exit the program")
        print("\nQUERY EXAMPLES:")
        print("  • What is machine learning?")
        print("  • How to learn Python programming?")
        print("  • Latest news about artificial intelligence")
        print("  • Best practices for web development")
        print("\nFEATURES:")
        print("  🤖 AI-powered responses using multiple LLM providers")
        print("  🔍 Advanced web scraping and content extraction")
        print("  🧠 ML-powered semantic similarity matching")
        print("  💾 Intelligent caching to avoid redundant searches")
        print("  🔄 Graceful fallbacks when APIs are unavailable")
        print("\nAPI KEYS (optional):")
        print("  Set environment variables for enhanced features:")
        print("  • OPENAI_API_KEY - For GPT models")
        print("  • GROQ_API_KEY - For fast inference")
        print("  • GOOGLE_API_KEY - For Gemini models")
        print("  • HUGGINGFACE_API_KEY - For HF models")
    
    def _print_stats(self):
        """Print detailed system statistics."""
        
        stats = self.engine.get_stats()
        
        print("\n📊 DETAILED STATISTICS")
        print("="*40)
        print(f"System Mode: {'Advanced' if stats.get('advanced_mode') else 'Basic'}")
        print(f"ML Embeddings: {'✅' if stats.get('ml_mode') else '❌'}")
        print(f"LLM Available: {'✅' if stats.get('llm_available') else '❌'}")
        print(f"Web Scraping: {'✅' if stats.get('web_scraping') else '❌'}")
        print()
        print(f"Total Queries Processed: {stats['total_queries']}")
        print(f"Cached Responses: {stats['cached_responses']}")
        print(f"Cache Hit Rate: {(stats['cached_responses']/max(stats['total_queries'], 1)*100):.1f}%")
        print(f"Vector Dimension: {stats['vector_dimension']}")
        print(f"FAISS Index Entries: {stats.get('faiss_entries', 0)}")
        print(f"Similarity Threshold: {stats['similarity_threshold']}")
        
        if stats.get('available_llm_providers'):
            print(f"\nAvailable LLM Providers:")
            for provider in stats['available_llm_providers']:
                print(f"  • {provider.upper()}")
        
        if stats['most_accessed_queries']:
            print(f"\n🔥 MOST POPULAR QUERIES:")
            for query, count in stats['most_accessed_queries'][:5]:
                print(f"  • {query} ({count} times)")


async def main():
    """Main entry point."""
    
    cli = FinalRipplicaCLI()
    
    if len(sys.argv) > 1:
        # Command line query
        query = ' '.join(sys.argv[1:])
        await cli.run_query(query)
    else:
        # Interactive mode
        await cli.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Thank you for using Ripplica! Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)