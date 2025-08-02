# 🏗️ Ripplica Query Agent Architecture

## 📋 System Overview

The Ripplica Query Agent is a full-stack AI-powered web search system that processes user queries, validates them, checks for cached similar results, and performs intelligent web scraping with LLM-powered summarization.

## 🔧 Core Components

### 1. **Query Processing Engine** (`src/query_processor.py`)
- **Query Validation**: Classifies queries as valid/invalid
- **Query Cleaning**: Normalizes and preprocesses queries
- **Intent Classification**: Categorizes query types (factual, comparison, how-to, etc.)

### 2. **Vector Store System** (`src/working_vector_store.py`)
- **FAISS Integration**: High-performance vector similarity search
- **Embedding Generation**: Sentence-transformers for semantic understanding
- **Cache Management**: Stores and retrieves similar queries
- **Similarity Matching**: Finds cached responses for similar queries (threshold: 0.7)

### 3. **Web Scraping Engine** (`src/working_web_scraper.py`)
- **Playwright Integration**: Headless browser automation
- **Multi-Engine Support**: Google (primary) + DuckDuckGo (fallback)
- **Content Extraction**: BeautifulSoup for clean text extraction
- **Error Handling**: Graceful degradation when scraping fails

### 4. **LLM Integration** (`src/llm_providers.py`)
- **Multiple Providers**: OpenAI, Groq, Google Gemini, Hugging Face
- **Fallback System**: Automatic provider switching on failure
- **Response Generation**: Intelligent summarization of scraped content

### 5. **Web Frameworks**
- **FastAPI Version** (`fastapi_app.py`): Modern async API with automatic docs
- **Flask Version** (`flask_app.py`): Traditional web framework alternative

## 🔄 System Flow

```
User Query Input
       ↓
1. Query Validation
   ├─ Invalid → Return error message
   └─ Valid → Continue
       ↓
2. Vector Similarity Search
   ├─ Similar found (>0.7) → Return cached result
   └─ No match → Continue
       ↓
3. Web Search & Scraping
   ├─ Google Search → Extract URLs
   ├─ Fallback to DuckDuckGo if needed
   └─ Scrape top 5 websites
       ↓
4. LLM Processing
   ├─ Summarize scraped content
   ├─ Generate coherent response
   └─ Try multiple providers if needed
       ↓
5. Cache & Return
   ├─ Store in vector database
   └─ Return to user
```

## 🗂️ Data Models (`src/models.py`)

### **QueryRequest**
```python
{
    "query": str,
    "search_engine": "google|duckduckgo",
    "llm_provider": "openai|groq|google|huggingface",
    "use_cache": bool,
    "max_results": int
}
```

### **CachedResponse**
```python
{
    "query": str,
    "response": str,
    "sources": List[str],
    "confidence_score": float,
    "llm_provider": str,
    "created_at": datetime,
    "access_count": int
}
```

## 🚀 API Endpoints

### **Web Interface**
- `GET /` - Main HTML interface
- `POST /api/query` - Process user queries
- `GET /api/health` - System health check

### **FastAPI Additional**
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API docs

## 🧠 AI/ML Components

### **Embedding Model**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: Convert queries to 384-dimensional vectors
- **Use Case**: Semantic similarity matching

### **Vector Database**
- **Engine**: FAISS (Facebook AI Similarity Search)
- **Index Type**: Flat L2 distance
- **Storage**: Persistent JSON + binary index files

### **LLM Providers**
1. **OpenAI GPT-3.5/4** - Primary choice for quality
2. **Groq Mixtral** - Fast inference (model deprecated, needs update)
3. **Google Gemini** - Free tier available
4. **Hugging Face** - Open source models

## 🔒 Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_KEY=your_hf_key
DEFAULT_LLM_PROVIDER=huggingface
CACHE_SIZE=1000
SIMILARITY_THRESHOLD=0.8
MAX_SEARCH_RESULTS=5
```

## 📊 Performance Features

### **Caching Strategy**
- **Exact Match**: O(1) lookup for identical queries
- **Semantic Match**: Vector similarity for related queries
- **LRU Eviction**: Automatic cache management
- **Persistent Storage**: Survives application restarts

### **Error Handling**
- **Graceful Degradation**: System works even if components fail
- **Provider Fallbacks**: Multiple LLM providers for reliability
- **Search Engine Fallback**: DuckDuckGo when Google fails
- **Timeout Management**: Prevents hanging requests

### **Scalability**
- **Async Processing**: Non-blocking query handling (FastAPI)
- **Batch Operations**: Efficient vector operations
- **Memory Management**: Configurable cache sizes
- **Resource Cleanup**: Proper browser session management

## 🛠️ Development Setup

### **Requirements**
- Python 3.8+
- Playwright browsers
- FAISS library
- Sentence transformers
- Web framework (FastAPI/Flask)

### **Installation**
```bash
pip install -r requirements_fastapi.txt  # or requirements.txt
playwright install chromium
```

### **Running**
```bash
# FastAPI version
python start_server.py

# Flask version  
python start_flask.py

# CLI version
python final_ripplica_cli.py
```

## 🎯 Key Engineering Decisions

### **Why FAISS over ChromaDB/Pinecone?**
- **Performance**: Faster for small-medium datasets
- **Local**: No external dependencies
- **Cost**: Free and self-hosted

### **Why Multiple LLM Providers?**
- **Reliability**: Fallback when one provider fails
- **Cost Optimization**: Use free tiers effectively
- **Quality**: Different models for different query types

### **Why Playwright over Requests?**
- **JavaScript Rendering**: Handles dynamic content
- **Anti-Bot Evasion**: More human-like browsing
- **Reliability**: Better success rates

### **Why Both FastAPI and Flask?**
- **FastAPI**: Modern, async, automatic docs
- **Flask**: Simpler, more familiar, synchronous
- **Choice**: Different deployment preferences

## 📈 Future Enhancements

1. **Advanced Caching**: Redis for distributed caching
2. **Query Analytics**: Track popular queries and performance
3. **Multi-language**: Support for non-English queries
4. **Real-time Updates**: WebSocket for live results
5. **Advanced Filtering**: Date ranges, source filtering
6. **User Accounts**: Personalized query history
7. **API Rate Limiting**: Prevent abuse
8. **Monitoring**: Logging and metrics dashboard
