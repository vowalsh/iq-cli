"""
Web search functionality for IQ CLI.
Handles fetching search results from SerpAPI.
"""

import requests
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class SearchResult:
    """Represents a single search result."""
    
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet
    
    def __repr__(self):
        return f"SearchResult(title='{self.title[:50]}...', url='{self.url}')"


class WebSearcher:
    """Handles web search operations using SerpAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SerpAPI key not found. Set SERPAPI_KEY environment variable.")
        
        self.base_url = "https://serpapi.com/search"
    
    def search(self, query: str, num_results: int = 8) -> List[SearchResult]:
        """
        Perform web search and return formatted results.
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: 8)
            
        Returns:
            List of SearchResult objects
        """
        params = {
            "q": query,
            "engine": "google",
            "api_key": self.api_key,
            "num": num_results,
            "gl": "us",  # Country
            "hl": "en"   # Language
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract organic results
            organic_results = data.get("organic_results", [])
            
            for result in organic_results[:num_results]:
                title = result.get("title", "")
                url = result.get("link", "")
                snippet = result.get("snippet", "")
                
                if title and url and snippet:
                    results.append(SearchResult(title, url, snippet))
            
            return results
            
        except requests.RequestException as e:
            raise Exception(f"Search request failed: {e}")
        except Exception as e:
            raise Exception(f"Search processing failed: {e}")
    
    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """
        Format search results for LLM consumption.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string with numbered results
        """
        if not results:
            return "No search results found."
        
        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"[{i}] {result.title}\n"
            formatted += f"URL: {result.url}\n"
            formatted += f"Snippet: {result.snippet}\n\n"
        
        return formatted
