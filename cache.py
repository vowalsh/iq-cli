"""
Cache management for IQ CLI.
Handles storing and retrieving Q&A pairs for offline access and quick retrieval.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from search import SearchResult


@dataclass
class CachedQA:
    """Represents a cached question-answer pair."""
    question: str
    answer: str
    search_results: List[Dict]  # Serialized SearchResult objects
    timestamp: str
    query_hash: str


class QACache:
    """Manages caching and retrieval of Q&A pairs."""
    
    def __init__(self, cache_dir: str = None, max_entries: int = 1000, expire_days: int = 30):
        """
        Initialize the Q&A cache.

        Args:
            cache_dir: Directory to store cache files (defaults to ~/.iq_cache)
            max_entries: Maximum number of entries to keep in cache
            expire_days: Number of days after which entries expire
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.iq_cache")
        self.cache_file = os.path.join(self.cache_dir, "qa_cache.json")
        self.max_entries = max_entries
        self.expire_days = expire_days
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache
        self.cache: List[CachedQA] = self._load_cache()
    
    def _load_cache(self) -> List[CachedQA]:
        """Load cache from disk."""
        if not os.path.exists(self.cache_file):
            return []
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [CachedQA(**item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            # If cache is corrupted, start fresh
            return []
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(qa) for qa in self.cache], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for the query for quick lookups."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _clean_expired_entries(self):
        """Remove expired entries from cache."""
        cutoff_date = datetime.now() - timedelta(days=self.expire_days)
        
        self.cache = [
            qa for qa in self.cache
            if datetime.fromisoformat(qa.timestamp) > cutoff_date
        ]
    
    def _similarity_score(self, query1: str, query2: str) -> float:
        """Calculate similarity score between two queries."""
        return SequenceMatcher(None, query1.lower().strip(), query2.lower().strip()).ratio()
    
    def store_qa(self, question: str, answer: str, search_results: List[SearchResult]):
        """
        Store a Q&A pair in the cache.
        
        Args:
            question: The original question
            answer: The generated answer
            search_results: List of SearchResult objects used
        """
        # Convert SearchResult objects to dictionaries for JSON serialization
        serialized_results = []
        for result in search_results:
            serialized_results.append({
                'title': str(result.title) if hasattr(result, 'title') else 'Unknown',
                'url': str(result.url) if hasattr(result, 'url') else '',
                'snippet': str(result.snippet) if hasattr(result, 'snippet') else '',
                'source': str(getattr(result, 'source', 'unknown'))
            })
        
        cached_qa = CachedQA(
            question=question,
            answer=answer,
            search_results=serialized_results,
            timestamp=datetime.now().isoformat(),
            query_hash=self._generate_query_hash(question)
        )
        
        # Remove any existing entry with the same hash
        self.cache = [qa for qa in self.cache if qa.query_hash != cached_qa.query_hash]
        
        # Add new entry at the beginning
        self.cache.insert(0, cached_qa)
        
        # Clean up old entries
        self._clean_expired_entries()
        
        # Limit cache size
        if len(self.cache) > self.max_entries:
            self.cache = self.cache[:self.max_entries]
        
        # Save to disk
        self._save_cache()
    
    def find_exact_match(self, query: str) -> Optional[CachedQA]:
        """
        Find an exact match for the query.
        
        Args:
            query: The search query
            
        Returns:
            CachedQA object if found, None otherwise
        """
        query_hash = self._generate_query_hash(query)
        
        for qa in self.cache:
            if qa.query_hash == query_hash:
                return qa
        
        return None
    
    def find_similar_questions(self, query: str, min_similarity: float = 0.7, max_results: int = 5) -> List[Tuple[CachedQA, float]]:
        """
        Find similar questions in the cache.
        
        Args:
            query: The search query
            min_similarity: Minimum similarity score (0.0 to 1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (CachedQA, similarity_score) sorted by similarity
        """
        similar_questions = []
        
        for qa in self.cache:
            similarity = self._similarity_score(query, qa.question)
            if similarity >= min_similarity:
                similar_questions.append((qa, similarity))
        
        # Sort by similarity (highest first) and limit results
        similar_questions.sort(key=lambda x: x[1], reverse=True)
        return similar_questions[:max_results]
    
    def search_cache(self, search_term: str, max_results: int = 10) -> List[CachedQA]:
        """
        Search through cached Q&As for a term.
        
        Args:
            search_term: Term to search for in questions and answers
            max_results: Maximum number of results to return
            
        Returns:
            List of matching CachedQA objects
        """
        search_term_lower = search_term.lower()
        matches = []
        
        for qa in self.cache:
            if (search_term_lower in qa.question.lower() or 
                search_term_lower in qa.answer.lower()):
                matches.append(qa)
        
        return matches[:max_results]
    
    def get_recent_entries(self, count: int = 10) -> List[CachedQA]:
        """
        Get the most recent cache entries.
        
        Args:
            count: Number of entries to return
            
        Returns:
            List of recent CachedQA objects
        """
        return self.cache[:count]
    
    def clear_cache(self):
        """Clear all cached entries."""
        self.cache = []
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        if not self.cache:
            return {
                'total_entries': 0,
                'oldest_entry': None,
                'newest_entry': None,
                'cache_size_mb': 0
            }
        
        cache_size = 0
        if os.path.exists(self.cache_file):
            cache_size = os.path.getsize(self.cache_file) / (1024 * 1024)  # MB
        
        return {
            'total_entries': len(self.cache),
            'oldest_entry': self.cache[-1].timestamp if self.cache else None,
            'newest_entry': self.cache[0].timestamp if self.cache else None,
            'cache_size_mb': round(cache_size, 2)
        }
