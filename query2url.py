import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
import json
from duckduckgo_search import DDGS
from googlesearch import search as google_search
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import urllib3
import asyncio
import urllib.parse
import random

load_dotenv()

# Disable SSL warnings for proxy mode
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# User agent strings for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

@dataclass
class SearchResult:
    """Represents a search result from any provider"""
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


class SearchProviderError(Exception):
    """Custom exception for search provider errors"""
    pass


class RateLimitError(SearchProviderError):
    """Raised when a provider is rate limited"""
    pass


class BaseSearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, name: str, rate_limit_delay: float = 1.0):
        self.name = name
        self.last_request_time = 0
        self.rate_limit_delay = rate_limit_delay
        self.backoff_until = 0  # For 429 backoff handling
    
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[str]:
        """Search for URLs using the provider"""
        pass
    
    def _rate_limit_check(self):
        """Ensure rate limiting is respected"""
        # Check if we're in backoff period
        if time.time() < self.backoff_until:
            remaining = self.backoff_until - time.time()
            print(f"[INFO] {self.name} in backoff period, waiting {remaining:.1f} seconds")
            time.sleep(remaining)
        
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _handle_429_backoff(self):
        """Handle 429 rate limit with 32 second backoff"""
        self.backoff_until = time.time() + 32
        print(f"[WARNING] {self.name} received 429, backing off for 32 seconds")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string"""
        return random.choice(USER_AGENTS)
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Common request handling with error checking"""
        self._rate_limit_check()
        
        try:
            response = requests.request(method, url, timeout=kwargs.get('timeout', 10), **kwargs)
            
            if response.status_code == 429:
                self._handle_429_backoff()
                raise RateLimitError(f"{self.name} rate limit exceeded")
            elif response.status_code != 200:
                raise SearchProviderError(f"{self.name} error: {response.status_code}")
            
            return response
        except requests.RequestException as e:
            raise SearchProviderError(f"{self.name} request failed: {str(e)}")
    
    def _extract_urls_from_json(self, data: dict, url_paths: List[Tuple[str, str]], num_results: int) -> List[str]:
        """Extract URLs from JSON response using multiple possible paths"""
        urls = []
        
        for container_path, url_key in url_paths:
            container = data
            for key in container_path.split('.') if container_path else []:
                container = container.get(key, {})
            
            if isinstance(container, list):
                for item in container:
                    if url_key in item:
                        urls.append(item[url_key])
                        if len(urls) >= num_results:
                            return urls
            elif isinstance(container, dict) and url_key in container:
                urls.append(container[url_key])
        
        return urls


class Crawl4AIGoogleProvider(BaseSearchProvider):
    """Crawl4AI-based Google search provider"""
    
    def __init__(self):
        super().__init__("Crawl4AI-Google", 1.2)  # Updated rate limit to 1.2 seconds
    
    async def _google_search_async(self, query: str, start_page: int = 0) -> str:
        """Perform async Google search using crawl4ai"""
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        
        # Construct Google search URL with pagination
        if start_page > 0:
            google_url = f"https://www.google.com/search?q={encoded_query}&start={start_page}"
        else:
            google_url = f"https://www.google.com/search?q={encoded_query}"
        
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=google_url)
            return result.markdown
    
    def _extract_urls_from_markdown(self, markdown_content: str) -> List[str]:
        """Extract URLs from Google search results markdown - ONLY main blue links"""
        
        # URLs to exclude (Google's internal links and artifacts)
        exclude_patterns = [
            r'google\.com/search',           # Google search pages
            r'google\.com/setprefs',         # Google preferences
            r'google\.com/webhp',            # Google homepage
            r'google\.de/intl',              # Google international
            r'accounts\.google\.com',        # Google accounts
            r'policies\.google\.com',        # Google policies
            r'support\.google\.com',         # Google support
            r'translate\.google\.com',       # Google translate
            r'fonts\.gstatic\.com',          # Google fonts
            r'google\.com/preferences',      # Google preferences
            r'[?&]ved=',                     # Google tracking parameters
            r'[?&]sa=',                      # Google search parameters
            r'[?&]ei=',                      # Google tracking
            r'[?&]start=',                   # Google pagination
            r'[?&]tbm=',                     # Google search type
            r'[?&]udm=',                     # Google search mode
            r'[?&]source=',                  # Google source tracking
            r'[?&]hl=',                      # Language parameters (any language)
            r'[?&]client=search',            # Google client parameter
        ]
        
        main_urls = []
        seen_urls = set()
        
        # ONLY extract main result URLs from ### lines (blue links)
        # The actual target URL is typically the LAST URL in the ### line
        main_result_pattern = r'###.*?\((https?://[^\s\)]+)\)'
        
        # Debug: Look for ### patterns
        debug_lines = [line for line in markdown_content.split('\n') if '###' in line and 'http' in line]
        print(f"[DEBUG] Found {len(debug_lines)} ### lines with URLs")
        if debug_lines:
            print(f"[DEBUG] Sample ### line: {debug_lines[0][:150]}...")
            
        # Also debug what the regex actually finds
        test_matches = re.findall(main_result_pattern, markdown_content, re.MULTILINE | re.IGNORECASE)
        print(f"[DEBUG] Regex found {len(test_matches)} main URLs: {test_matches[:3]}")
        
        # Debug: Let's see some lines around the ### to understand the format
        lines = markdown_content.split('\n')
        for i, line in enumerate(lines):
            if '###' in line and 'Papers With Code' in line:
                print(f"[DEBUG] Found target line {i}: {line}")
                if i > 0:
                    print(f"[DEBUG] Line {i-1}: {lines[i-1]}")
                if i < len(lines) - 1:
                    print(f"[DEBUG] Line {i+1}: {lines[i+1]}")
                break
        
        # Process each ### line separately to find the main URL
        lines = markdown_content.split('\n')
        for line in lines:
            if '###' not in line:
                continue
                
            # Find all URLs in this ### line
            line_urls = re.findall(r'\((https?://[^\s\)]+)\)', line)
            if not line_urls:
                continue
                
            # Take the last non-Google URL from this line (the actual target)
            target_url = None
            for url in reversed(line_urls):  # Check from last to first
                url = re.sub(r'[,;.\s]+$', '', url)
                
                # Check if this is a Google URL we should skip
                is_google = False
                for exclude_pattern in exclude_patterns:
                    if re.search(exclude_pattern, url, re.IGNORECASE):
                        is_google = True
                        break
                
                if not is_google and url not in seen_urls:
                    target_url = url
                    break
            
            if target_url:
                seen_urls.add(target_url)
                main_urls.append(target_url)
                print(f"[DEBUG] Found main URL: {target_url}")
        
        # ONLY return main blue links - no other URLs
        return main_urls
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        """Search for URLs using crawl4ai Google search with pagination"""
        self._rate_limit_check()
        
        try:
            all_urls = []
            page = 0
            max_pages = 5  # Limit to 5 pages to avoid excessive requests
            
            while len(all_urls) < num_results and page < max_pages:
                start_index = page * 10  # Google typically shows 10 results per page
                
                print(f"[DEBUG] Searching page {page + 1} (start={start_index}) for query: {query}")
                
                # Run the async search for current page
                try:
                    # Try to get current event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, we need to run in a separate thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, self._google_search_async(query, start_index))
                            markdown_content = future.result()
                    else:
                        # Loop is not running, we can use it directly
                        markdown_content = loop.run_until_complete(self._google_search_async(query, start_index))
                except RuntimeError:
                    # No event loop, create a new one
                    markdown_content = asyncio.run(self._google_search_async(query, start_index))
                
                # Extract URLs from the markdown content
                page_urls = self._extract_urls_from_markdown(markdown_content)
                
                if not page_urls:
                    print(f"[DEBUG] No URLs found on page {page + 1}, stopping pagination")
                    break
                
                # Add new URLs (avoid duplicates)
                new_urls = []
                for url in page_urls:
                    if url not in all_urls:
                        new_urls.append(url)
                        all_urls.append(url)
                
                print(f"[DEBUG] Page {page + 1}: Found {len(page_urls)} URLs, {len(new_urls)} new URLs. Total: {len(all_urls)}")
                
                # If no new URLs found, stop pagination
                if not new_urls:
                    print(f"[DEBUG] No new URLs on page {page + 1}, stopping pagination")
                    break
                
                page += 1
                
                # Add delay between pages to be respectful
                if page < max_pages and len(all_urls) < num_results:
                    time.sleep(1.5)  # 1.5 second delay between pages
            
            if not all_urls:
                raise SearchProviderError("No results found from Crawl4AI Google search")
            
            print(f"[DEBUG] Total URLs collected: {len(all_urls)} from {page} pages")
            
            # Return up to num_results URLs
            return all_urls[:num_results]
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                raise RateLimitError(f"Crawl4AI Google rate limit: {e}")
            raise SearchProviderError(f"Crawl4AI Google search failed: {str(e)}")


class APISearchProvider(BaseSearchProvider):
    """Base class for API-based search providers"""
    
    def __init__(self, name: str, api_key_env: str, base_url: str, rate_limit_delay: float = 1.0):
        super().__init__(name, rate_limit_delay)
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url
        
        if not self.api_key:
            raise SearchProviderError(f"{api_key_env} not found in environment variables")
        
class HTMLScrapingProvider(BaseSearchProvider):
    """Base class for HTML scraping providers"""
    
    def _extract_google_urls(self, html: str, num_results: int) -> List[str]:
        """Extract URLs from Google search results HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Handle modern Google URL patterns
                if href.startswith('/url?') or href.startswith('https://www.google.com/url'):
                    # Extract URL from various Google redirect patterns
                    match = re.search(r'[&?](q|url)=([^&]+)', href)
                    if match:
                        actual_url = requests.utils.unquote(match.group(2))
                        if actual_url.startswith('http'):
                            urls.append(actual_url)
                            if len(urls) >= num_results:
                                break
                
                # Direct HTTP links (some modern Google results)
                elif href.startswith('http') and 'google.com' not in href:
                    urls.append(href)
                    if len(urls) >= num_results:
                        break
                        
            return urls
        except ImportError:
            raise SearchProviderError("BeautifulSoup4 required for HTML scraping providers")


class SerperProvider(APISearchProvider):
    """Serper API provider"""
    
    def __init__(self):
        super().__init__("Serper", "SERPER_API_KEY", "https://google.serper.dev/search", 0.6)  # Updated rate limit to 0.6 seconds
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        response = self._make_request('POST', self.base_url, 
            headers={'X-API-KEY': self.api_key, 'Content-Type': 'application/json'},
            json={'q': query, 'num': min(num_results, 100)}
        )
        
        urls = self._extract_urls_from_json(response.json(), [('organic', 'link')], num_results)
        if not urls:
            raise SearchProviderError("No results found from Serper API")
        return urls


class BraveProvider(APISearchProvider):
    """Brave Search API provider"""
    
    def __init__(self):
        super().__init__("Brave", "BRAVE_API_KEY", "https://api.search.brave.com/res/v1/web/search", 1.2)  # Updated rate limit to 1.2 seconds
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        response = self._make_request('GET', self.base_url,
            headers={'X-Subscription-Token': self.api_key, 'Accept': 'application/json'},
            params={'q': query, 'count': min(num_results, 20)}
        )
        
        urls = self._extract_urls_from_json(response.json(), [('web.results', 'url')], num_results)
        if not urls:
            raise SearchProviderError("No results found from Brave API")
        return urls


class SerpstackProvider(APISearchProvider):
    """Serpstack API provider"""
    
    def __init__(self):
        super().__init__("Serpstack", "SERPSTACK_API_KEY", "http://api.serpstack.com/search", 0.1)  # No rate limits but keep low for concurrency
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        response = self._make_request('GET', self.base_url,
            params={'access_key': self.api_key, 'query': query, 'num': min(num_results, 100)}
        )
        
        data = response.json()
        if 'error' in data:
            error_msg = data['error'].get('info', 'Unknown error')
            if 'rate limit' in error_msg.lower():
                raise RateLimitError(f"Serpstack rate limit: {error_msg}")
            raise SearchProviderError(f"Serpstack API error: {error_msg}")
        
        urls = self._extract_urls_from_json(data, [('organic_results', 'url')], num_results)
        if not urls:
            raise SearchProviderError("No results found from Serpstack")
        return urls


class ScrapingAntProvider(HTMLScrapingProvider):
    """ScrapingAnt provider using official client (free plan compatible)"""
    
    def __init__(self):
        super().__init__("ScrapingAnt", 0.1)  # No rate limits but keep low for concurrency
        self.api_key = os.getenv("SCRAPINGANT_API_KEY")
        
        if not self.api_key:
            raise SearchProviderError("SCRAPINGANT_API_KEY not found in environment variables")
        
        try:
            from scrapingant_client import ScrapingAntClient
            self.client = ScrapingAntClient(token=self.api_key)
        except ImportError:
            raise SearchProviderError("scrapingant-client not installed. Run: pip install scrapingant-client")
        except Exception as e:
            raise SearchProviderError(f"Failed to initialize ScrapingAnt client: {e}")
    
    def _build_search_url(self, query: str, search_engine: str = 'yahoo', num_results: int = 10) -> str:
        """Build search URL for different engines (free plan compatible)"""
        from urllib.parse import urlencode
        
        if search_engine == 'bing':
            params = {
                'q': query,
                'count': min(num_results, 50),
                'form': 'QBLH',
                'sp': '-1',
                'lq': '0',
                'pq': query,
                'qs': 'n',
                'sk': '',
                'cvid': 'A1B2C3D4E5F6789012345678901234567'  # Dummy CVID
            }
            return f"https://www.bing.com/search?{urlencode(params)}"
            
        elif search_engine == 'duckduckgo':
            params = {
                't': 'h_',
                'q': query,
                'ia': 'web'
            }
            return f"https://duckduckgo.com/?{urlencode(params)}"
            
        elif search_engine == 'yahoo':
            params = {
                'p': query,
                'fr': 'yfp-t',
                'fr2': 'p:fp,m:sb',
                'ei': 'UTF-8',
                'fp': '1'
            }
            return f"https://search.yahoo.com/search?{urlencode(params)}"
            
        else:
            # Default to Bing for better reliability
            params = {
                'q': query,
                'count': min(num_results, 50),
                'form': 'QBLH'
            }
            return f"https://www.bing.com/search?{urlencode(params)}"
    
    def _extract_search_urls(self, html_content: str, search_engine: str) -> List[str]:
        """Extract URLs from search results HTML with engine-specific selectors"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            urls = []
            
            if search_engine == 'bing':
                # Bing-specific selectors
                search_results = soup.find_all('li', class_='b_algo')
                for result in search_results:
                    link_elem = result.find('a', href=True)
                    if link_elem and link_elem['href'].startswith('http'):
                        urls.append(link_elem['href'])
            
            elif search_engine == 'duckduckgo':
                # DuckDuckGo-specific selectors with multiple fallbacks
                selectors = [
                    {'tag': 'div', 'attrs': {'data-testid': 'result'}},
                    {'tag': 'div', 'attrs': {'class': 'nrn-react-div'}},
                    {'tag': 'div', 'attrs': {'class': lambda x: x and 'result' in x.lower() if x else False}}
                ]
                
                for selector in selectors:
                    search_results = soup.find_all(selector['tag'], selector['attrs'])
                    if search_results:
                        break
                
                for result in search_results:
                    link_elem = result.find('a', href=True)
                    if link_elem:
                        href = link_elem['href']
                        # Clean DuckDuckGo redirect URLs
                        if href.startswith('/l/?uddg='):
                            try:
                                href = requests.utils.unquote(href.split('uddg=')[1])
                            except:
                                pass
                        if href.startswith('http'):
                            urls.append(href)
            
            elif search_engine == 'yahoo':
                # Yahoo-specific selectors
                search_results = soup.find_all('div', class_='algo')
                for result in search_results:
                    link_elem = result.find('a', href=True)
                    if link_elem and link_elem['href'].startswith('http'):
                        urls.append(link_elem['href'])
            
            # Generic fallback if no engine-specific results found
            if not urls:
                print(f"[WARNING] No results with {search_engine} selectors, trying generic approach")
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    href = link['href']
                    if (href.startswith('http') and 
                        'google.com' not in href and 
                        not any(skip in href.lower() for skip in ['privacy', 'terms', 'about', 'help'])):
                        urls.append(href)
                        if len(urls) >= 20:  # Limit generic fallback
                            break
            
            return urls
            
        except Exception as e:
            print(f"[ERROR] Error extracting URLs from {search_engine}: {e}")
            return []
    
    def _is_search_engine_artifact(self, url: str) -> bool:
        """Check if URL is a search engine artifact/internal link that should be filtered out"""
        url_lower = url.lower()
        
        # DuckDuckGo artifacts
        duckduckgo_artifacts = [
            'duckduckgo.com/mac',
            'duckduckgo.com/y.js',
            'duckduckgo.com/l/?',
            'duckduckgo.com/privacy',
            'duckduckgo.com/about',
            'duckduckgo.com/settings',
            'origin=funnel_browser',
            'ad_provider=',
            'click_metadata=',
            'ad_domain='
        ]
        
        # Yahoo artifacts  
        yahoo_artifacts = [
            'mail.yahoo.com',
            'login.yahoo.com',
            'chat.yahoo.com',
            'finance.yahoo.com',
            'sports.yahoo.com',
            'news.yahoo.com'
        ]
        
        # Bing artifacts
        bing_artifacts = [
            'bing.com/search?',
            'msn.com/play',
            'microsoft.com/bing',
            'go.microsoft.com/fwlink'
        ]
        
        # Check all artifact patterns
        all_artifacts = duckduckgo_artifacts + yahoo_artifacts + bing_artifacts
        
        for artifact in all_artifacts:
            if artifact in url_lower:
                return True
                
        return False
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        """Search using ScrapingAnt client with fallback between search engines"""
        self._rate_limit_check()
        
        # Try search engines in order with fallback (free plan compatible)
        search_engines = ['duckduckgo', 'bing', 'yahoo']
        
        for search_engine in search_engines:
            try:
                search_url = self._build_search_url(query, search_engine, num_results)
                print(f"[INFO] ScrapingAnt trying {search_engine}: {search_url}")
                
                # Use ScrapingAnt client general_request method
                result = self.client.general_request(
                    url=search_url,
                    proxy_type='datacenter',  # Use datacenter for free plan
                    proxy_country='US'
                )
                
                if result.status_code == 200:
                    urls = self._extract_search_urls(result.content, search_engine)
                    
                    # Simple validation and collect valid URLs
                    valid_urls = []
                    for url in urls:
                        if (url and isinstance(url, str) and 
                            url.startswith(('http://', 'https://')) and
                            len(url) > 10 and
                            not self._is_search_engine_artifact(url)):
                            valid_urls.append(url)
                    
                    # Remove duplicates while preserving order
                    unique_urls = list(dict.fromkeys(valid_urls))
                    
                    if unique_urls:
                        print(f"[INFO] ScrapingAnt {search_engine}: SUCCESS - found {len(unique_urls)} URLs")
                        return unique_urls[:num_results]
                    else:
                        print(f"[WARNING] ScrapingAnt {search_engine}: No valid URLs found, trying next engine")
                        
                elif result.status_code == 429:
                    print(f"[WARNING] ScrapingAnt {search_engine}: Rate limited (429), trying next engine")
                    
                else:
                    print(f"[WARNING] ScrapingAnt {search_engine}: HTTP {result.status_code}, trying next engine")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if any(term in error_msg for term in ["rate", "limit", "429"]):
                    print(f"[WARNING] ScrapingAnt {search_engine}: Rate limited - {e}")
                else:
                    print(f"[WARNING] ScrapingAnt {search_engine}: Failed - {e}")
                
                # Continue to next engine on any error
                continue
                
            # Small delay before trying next engine
            time.sleep(0.5)
        
        raise SearchProviderError("ScrapingAnt: All search engines failed or returned no results")


class ScrapingBeeProvider(APISearchProvider):
    """ScrapingBee Google Search API provider with structured results"""
    
    def __init__(self):
        try:
            super().__init__("ScrapingBee", "SCRAPINGBEE_API_KEY", "https://app.scrapingbee.com/api/v1/store/google", 0.1)  # No rate limits but keep low for concurrency
        except SearchProviderError:
            raise SearchProviderError("ScrapingBee API key not configured")
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        # Use ScrapingBee's dedicated Google Search API
        params = {
            'api_key': self.api_key,
            'search': query,
            'nb_results': min(num_results, 100),
            'country_code': 'us'
        }
        
        try:
            response = self._make_request('GET', self.base_url, params=params)
            
            # Check for ScrapingBee specific error responses
            if response.status_code == 400:
                raise SearchProviderError("ScrapingBee: Bad request - check parameters")
            elif response.status_code == 401:
                raise SearchProviderError("ScrapingBee: Invalid API key")
            elif response.status_code == 402:
                raise SearchProviderError("ScrapingBee: Insufficient credits")
            elif response.status_code == 422:
                raise SearchProviderError("ScrapingBee: Unable to process request")
            
            # Parse structured JSON response
            data = response.json()
            urls = []
            
            # Extract URLs from organic results
            organic_results = data.get('organic_results', [])
            for result in organic_results[:num_results]:
                url = result.get('url')
                if url:
                    urls.append(url)
            
            if not urls:
                raise SearchProviderError("No results found from ScrapingBee Google API")
            return urls
            
        except json.JSONDecodeError:
            raise SearchProviderError("ScrapingBee returned invalid JSON")
        except requests.RequestException as e:
            raise SearchProviderError(f"ScrapingBee request failed: {str(e)}")


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider"""
    
    def __init__(self):
        super().__init__("DuckDuckGo", 2.0)  # Updated rate limit to 2.0 seconds
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        self._rate_limit_check()
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
                
                urls = []
                for result in results:
                    for key in ['href', 'link', 'url']:
                        if key in result:
                            urls.append(result[key])
                            break
                
                if not urls:
                    raise SearchProviderError("No results found from DuckDuckGo")
                return urls[:num_results]
                
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                self._handle_429_backoff()
                raise RateLimitError("DuckDuckGo rate limit exceeded")
            raise SearchProviderError(f"DuckDuckGo search failed: {str(e)}")


class GoogleSearchPythonProvider(BaseSearchProvider):
    """googlesearch-python provider"""
    
    def __init__(self):
        super().__init__("GoogleSearch-Python", 2.0)  # Updated rate limit to 2.0 seconds
    
    def search(self, query: str, num_results: int = 10) -> List[str]:
        self._rate_limit_check()
        
        try:
            # Rotate user agent for each request
            user_agent = self._get_random_user_agent()
            
            urls = list(google_search(query, num_results=num_results, lang='en', user_agent=user_agent))
            
            if not urls:
                raise SearchProviderError("No results found from Google Search Python")
            return urls[:num_results]
            
        except Exception as e:
            if any(term in str(e).lower() for term in ["429", "blocked", "captcha"]):
                self._handle_429_backoff()
                raise RateLimitError("Google Search Python rate limit or blocking detected")
            raise SearchProviderError(f"Google Search Python failed: {str(e)}")



class MultiProviderSearchEngine:
    """Main search engine class that manages multiple providers with fallback logic"""
    
    def __init__(self, providers: Optional[List[BaseSearchProvider]] = None):
        self.providers = providers or self._initialize_default_providers()
        self.last_successful_provider = None
    
    def _initialize_default_providers(self) -> List[BaseSearchProvider]:
        """Initialize default providers in priority order"""
        # Updated provider order as requested:
        # 1. Crawl4ai (first)
        # 2. Brave (second)
        # 3. Serper (third)
        # 4. ScrapingAnt (fourth)
        # 5. DuckDuckGo (fifth)
        # 6. GoogleSearch-Python (sixth)
        # 7. ScrapingBee (1 before last)
        # 8. Serpstack (last)
        provider_classes = [
            Crawl4AIGoogleProvider,      # 1st - 1.2 secs
            BraveProvider,               # 2nd - 1.2 secs
            SerperProvider,              # 3rd - 0.6 secs
            ScrapingAntProvider,         # 4th - No rate limits
            DuckDuckGoProvider,          # 5th - 2 secs with user agent rotation
            GoogleSearchPythonProvider,  # 6th - 2 secs with user agent rotation
            ScrapingBeeProvider,         # 7th - No rate limits
            SerpstackProvider            # 8th (last) - No rate limits
        ]
        
        providers = []
        for provider_class in provider_classes:
            try:
                providers.append(provider_class())
            except SearchProviderError as e:
                print(f"[WARNING] Failed to initialize {provider_class.__name__}: {e}")
        
        if not providers:
            raise SearchProviderError("No search providers could be initialized")
        return providers
    
    def search(self, query: str, num_results: int = 10, max_retries: int = 2) -> List[str]:
        """Search using multiple providers with fallback logic"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        print(f"[INFO] Searching for: '{query}' (requesting {num_results} results)")
        
        # Try last successful provider first
        providers_to_try = self.providers.copy()
        if self.last_successful_provider and self.last_successful_provider in providers_to_try:
            providers_to_try.remove(self.last_successful_provider)
            providers_to_try.insert(0, self.last_successful_provider)
        
        last_error = None
        
        for provider in providers_to_try:
            for attempt in range(max_retries + 1):
                try:
                    print(f"[INFO] Trying {provider.name} (attempt {attempt + 1})")
                    urls = provider.search(query, num_results)
                    unique_urls = self._validate_and_deduplicate_urls(urls, num_results)
                    
                    if unique_urls:
                        self.last_successful_provider = provider
                        return unique_urls
                    else:
                        print(f"[WARNING] {provider.name} returned no valid URLs")
                        
                except RateLimitError as e:
                    print(f"[WARNING] {provider.name} rate limited: {e}")
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        print(f"[INFO] Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    last_error = e
                    break
                    
                except SearchProviderError as e:
                    print(f"[WARNING] {provider.name} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(1)
                    last_error = e
                    
                except Exception as e:
                    print(f"[ERROR] Unexpected error with {provider.name}: {e}")
                    last_error = SearchProviderError(f"Unexpected error: {e}")
                    break
        
        raise SearchProviderError(f"All search providers failed. Last error: {last_error}")
    
    def _validate_and_deduplicate_urls(self, urls: List[str], max_results: int) -> List[str]:
        """Validate and deduplicate URLs"""
        seen = set()
        valid_urls = []
        
        for url in urls:
            if not url or not isinstance(url, str):
                continue
                
            url = url.strip()
            if not url.startswith(('http://', 'https://')) or url in seen:
                continue
                
            seen.add(url)
            valid_urls.append(url)
            
            if len(valid_urls) >= max_results:
                break
        
        return valid_urls
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for provider in self.providers:
            provider_status = {
                'name': provider.name,
                'available': True,
                'last_request_time': provider.last_request_time,
                'rate_limit_delay': provider.rate_limit_delay
            }
            
            if hasattr(provider, 'api_key'):
                provider_status['has_api_key'] = bool(provider.api_key)
            if hasattr(provider, 'cse_id'):
                provider_status['has_cse_id'] = bool(provider.cse_id)
            
            status[provider.name] = provider_status
        
        return status
    
    def test_providers(self, test_query: str = "python programming") -> Dict[str, Any]:
        """Test all providers with a simple query"""
        results = {}
        
        for provider in self.providers:
            try:
                start_time = time.time()
                urls = provider.search(test_query, 3)
                end_time = time.time()
                
                results[provider.name] = {
                    'status': 'success',
                    'url_count': len(urls),
                    'response_time': round(end_time - start_time, 2),
                    'sample_urls': urls[:2]
                }
                
            except Exception as e:
                results[provider.name] = {
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        return results


def query2url(query: str, num_results: int = 10, providers: Optional[List[BaseSearchProvider]] = None) -> List[str]:
    """Convenience function for quick searches"""
    engine = MultiProviderSearchEngine(providers)
    return engine.search(query, num_results)