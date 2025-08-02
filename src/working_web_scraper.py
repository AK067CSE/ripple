"""Working web scraper that uses DuckDuckGo and fallback methods."""

import asyncio
import time
import aiohttp
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re

from .models import WebPage, SearchResult, SearchEngine
from .config import settings

# Try to import Playwright, fall back if not available
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class WorkingWebScraper:
    """Web scraper that works with or without Playwright."""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright_available = PLAYWRIGHT_AVAILABLE
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize browser and session."""
        # Always initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout),
            headers={'User-Agent': settings.user_agent}
        )
        
        # Try to initialize Playwright if available
        if self.playwright_available:
            try:
                playwright = await async_playwright().start()
                self.browser = await playwright.chromium.launch(headless=True)
                print("✅ Playwright browser initialized")
            except Exception as e:
                print(f"⚠️  Playwright initialization failed: {e}")
                self.playwright_available = False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
        if self.session:
            await self.session.close()
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search DuckDuckGo using their HTML interface."""
        
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            print(f"🔍 Searching DuckDuckGo: {search_url}")
            
            async with self.session.get(search_url) as response:
                print(f"📡 DuckDuckGo response status: {response.status}")
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    
                    # Find result links
                    result_links = soup.find_all('a', class_='result__a')
                    print(f"🔗 Found {len(result_links)} result links")
                    
                    for link in result_links[:max_results]:
                        try:
                            title = link.get_text(strip=True)
                            url = link.get('href')
                            
                            print(f"🔍 Raw result: title='{title[:30]}...', url='{url[:50]}...'")
                            
                            if url and title:
                                # Clean up the URL (DuckDuckGo sometimes wraps URLs)
                                if url.startswith('//duckduckgo.com/l/?uddg='):
                                    # Extract the actual URL from DuckDuckGo's redirect
                                    import urllib.parse
                                    try:
                                        # Parse the redirect URL
                                        parsed_url = urllib.parse.urlparse('https:' + url)
                                        query_params = urllib.parse.parse_qs(parsed_url.query)
                                        if 'uddg' in query_params:
                                            url = urllib.parse.unquote(query_params['uddg'][0])
                                            print(f"🔧 Cleaned URL: {url[:50]}...")
                                    except Exception as e:
                                        print(f"Error cleaning URL: {e}")
                                        continue
                                
                                # Also handle relative URLs
                                elif url.startswith('//'):
                                    url = 'https:' + url
                                elif url.startswith('/'):
                                    url = 'https://duckduckgo.com' + url
                                
                                if url.startswith('http'):
                                    results.append({
                                        'title': title,
                                        'url': url
                                    })
                                    print(f"✅ Added result: {title[:50]}...")
                                else:
                                    print(f"❌ Skipped non-http URL: {url[:50]}...")
                        except Exception as e:
                            print(f"Error parsing result: {e}")
                            continue
                    
                    print(f"📊 Total results found: {len(results)}")
                    return results
                else:
                    print(f"❌ DuckDuckGo returned status {response.status}")
                
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
        
        return []
    
    async def search_google_simple(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Simple Google search using requests (no JavaScript)."""
        
        try:
            search_url = f"https://www.google.com/search?q={query}&num={max_results}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    
                    # Try different selectors for Google results
                    selectors = [
                        'h3',  # Simple h3 tags
                        '.LC20lb',  # Google result title class
                        '.DKV0Md',  # Alternative title class
                    ]
                    
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            for element in elements[:max_results]:
                                try:
                                    title = element.get_text(strip=True)
                                    
                                    # Find parent link
                                    link = element.find_parent('a')
                                    if not link:
                                        link = element.find('a')
                                    
                                    if link and link.get('href'):
                                        url = link['href']
                                        
                                        # Clean Google redirect URLs
                                        if url.startswith('/url?q='):
                                            url = url.split('/url?q=')[1].split('&')[0]
                                            import urllib.parse
                                            url = urllib.parse.unquote(url)
                                        
                                        if url.startswith('http') and title:
                                            results.append({
                                                'title': title,
                                                'url': url
                                            })
                                            
                                            if len(results) >= max_results:
                                                break
                                except Exception as e:
                                    continue
                            
                            if results:
                                break
                    
                    return results
                
        except Exception as e:
            print(f"Error searching Google: {e}")
        
        return []
    
    async def search_and_scrape(
        self, 
        query: str, 
        search_engine: SearchEngine = SearchEngine.DUCKDUCKGO, 
        max_results: int = 5
    ) -> SearchResult:
        """Search and scrape web pages."""
        
        start_time = time.time()
        
        # Perform search
        if search_engine == SearchEngine.DUCKDUCKGO:
            search_results = await self.search_duckduckgo(query, max_results)
        else:
            search_results = await self.search_google_simple(query, max_results)
        
        # If no results, try the other search engine
        if not search_results:
            if search_engine == SearchEngine.DUCKDUCKGO:
                print("⚠️  DuckDuckGo failed, trying Google...")
                search_results = await self.search_google_simple(query, max_results)
            else:
                print("⚠️  Google failed, trying DuckDuckGo...")
                search_results = await self.search_duckduckgo(query, max_results)
        
        # Scrape the pages
        pages = []
        for result in search_results:
            try:
                page = await self.scrape_page(result['url'], result['title'])
                if page:
                    pages.append(page)
            except Exception as e:
                print(f"Error scraping {result['url']}: {e}")
                continue
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query=query,
            pages=pages,
            search_time=search_time,
            total_results=len(search_results),
            search_engine=search_engine
        )
    
    async def scrape_page(self, url: str, title: str = "") -> Optional[WebPage]:
        """Scrape content from a single web page."""
        
        try:
            # Try Playwright first if available
            if self.playwright_available and self.browser:
                return await self._scrape_with_playwright(url, title)
            else:
                return await self._scrape_with_requests(url, title)
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    async def _scrape_with_playwright(self, url: str, title: str) -> Optional[WebPage]:
        """Scrape using Playwright (handles JavaScript)."""
        
        page = await self.browser.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
            
            # Get page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text content
            text_content = self._extract_text_content(soup)
            
            # Get actual title if not provided
            if not title:
                title_element = soup.find('title')
                title = title_element.get_text(strip=True) if title_element else url
            
            return WebPage(
                url=url,
                title=title,
                content=text_content,
                word_count=len(text_content.split())
            )
            
        finally:
            await page.close()
    
    async def _scrape_with_requests(self, url: str, title: str) -> Optional[WebPage]:
        """Scrape using aiohttp (no JavaScript support)."""
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract text content
                    text_content = self._extract_text_content(soup)
                    
                    # Get actual title if not provided
                    if not title:
                        title_element = soup.find('title')
                        title = title_element.get_text(strip=True) if title_element else url
                    
                    return WebPage(
                        url=url,
                        title=title,
                        content=text_content,
                        word_count=len(text_content.split())
                    )
                    
        except Exception as e:
            print(f"Error with requests scraping: {e}")
            return None
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main',
            'article', 
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.container'
        ]
        
        main_content = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        # Limit length
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    async def get_page_content(self, url: str) -> Optional[str]:
        """Get raw content from a URL."""
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            print(f"Error getting content from {url}: {e}")
        
        return None