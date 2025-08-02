"""Web scraping module using Playwright and BeautifulSoup."""

import asyncio
import time
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
import re

from .models import WebPage, SearchResult, SearchEngine
from .config import settings


class WebScraper:
    """Handles web scraping operations."""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize browser and session."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout),
            headers={'User-Agent': settings.user_agent}
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
        if self.session:
            await self.session.close()
    
    async def search_google(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search Google and extract result URLs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title and URL
        """
        if not self.browser:
            await self.initialize()
        
        page = await self.browser.new_page()
        
        try:
            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={query}&num={max_results}"
            await page.goto(search_url, wait_until="networkidle")
            
            # Wait for results to load
            await page.wait_for_selector('div[data-ved]', timeout=10000)
            
            # Extract search results
            results = []
            result_elements = await page.query_selector_all('div[data-ved] h3')
            
            for element in result_elements[:max_results]:
                try:
                    # Get the title
                    title = await element.inner_text()
                    
                    # Get the parent link
                    link_element = await element.query_selector('xpath=ancestor::a')
                    if link_element:
                        url = await link_element.get_attribute('href')
                        if url and url.startswith('http'):
                            results.append({
                                'title': title.strip(),
                                'url': url
                            })
                except Exception as e:
                    print(f"Error extracting result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error searching Google: {e}")
            return []
        finally:
            await page.close()
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and extract result URLs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title and URL
        """
        if not self.session:
            await self.initialize()
        
        try:
            # DuckDuckGo instant answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get('https://api.duckduckgo.com/', params=params) as response:
                data = await response.json()
                
                results = []
                
                # Extract from related topics
                if 'RelatedTopics' in data:
                    for topic in data['RelatedTopics'][:max_results]:
                        if isinstance(topic, dict) and 'FirstURL' in topic:
                            results.append({
                                'title': topic.get('Text', '').split(' - ')[0],
                                'url': topic['FirstURL']
                            })
                
                # If not enough results, try web search
                if len(results) < max_results:
                    # Use DuckDuckGo HTML search as fallback
                    search_url = f"https://duckduckgo.com/html/?q={query}"
                    async with self.session.get(search_url) as response:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        for result in soup.find_all('a', class_='result__a')[:max_results]:
                            title = result.get_text().strip()
                            url = result.get('href')
                            if title and url:
                                results.append({
                                    'title': title,
                                    'url': url
                                })
                
                return results[:max_results]
                
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []
    
    async def scrape_webpage(self, url: str) -> Optional[WebPage]:
        """
        Scrape content from a webpage.
        
        Args:
            url: URL to scrape
            
        Returns:
            WebPage object or None if scraping failed
        """
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract main content
                content_selectors = [
                    'article', 'main', '.content', '#content', '.post', '.entry'
                ]
                
                content = ""
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content = content_elem.get_text()
                        break
                
                # Fallback to body content
                if not content:
                    body = soup.find('body')
                    if body:
                        content = body.get_text()
                
                # Clean content
                content = self._clean_text(content)
                
                if len(content) < 100:  # Skip pages with too little content
                    return None
                
                return WebPage(
                    url=url,
                    title=title,
                    content=content[:5000],  # Limit content length
                    word_count=len(content.split())
                )
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        return text.strip()
    
    async def search_and_scrape(
        self, 
        query: str, 
        search_engine: SearchEngine = SearchEngine.GOOGLE,
        max_results: int = 5
    ) -> SearchResult:
        """
        Search and scrape web pages for a query.
        
        Args:
            query: Search query
            search_engine: Search engine to use
            max_results: Maximum number of results to scrape
            
        Returns:
            SearchResult object
        """
        start_time = time.time()
        
        # Get search results
        if search_engine == SearchEngine.GOOGLE:
            search_results = await self.search_google(query, max_results)
        else:
            search_results = await self.search_duckduckgo(query, max_results)
        
        # Scrape pages concurrently
        scrape_tasks = []
        for result in search_results:
            task = self.scrape_webpage(result['url'])
            scrape_tasks.append(task)
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        
        async def scrape_with_semaphore(task):
            async with semaphore:
                return await task
        
        scraped_pages = await asyncio.gather(
            *[scrape_with_semaphore(task) for task in scrape_tasks],
            return_exceptions=True
        )
        
        # Filter successful scrapes
        pages = []
        for i, page in enumerate(scraped_pages):
            if isinstance(page, WebPage):
                pages.append(page)
            elif not isinstance(page, Exception):
                # Handle case where scraping returned None
                continue
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query=query,
            pages=pages,
            search_engine=search_engine,
            total_results=len(search_results),
            search_time=search_time
        )