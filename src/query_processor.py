"""Query processing and validation module."""

import re
import string
from typing import List, Tuple, Optional
from .models import QueryIntent


class QueryProcessor:
    """Handles query validation, cleaning, and intent classification."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what|who|when|where|which)\b',
                r'\b(fact|facts|information|data)\b',
                r'\b(is|are|was|were|did|does|do)\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(vs|versus|compare|comparison|difference|better|best|worst)\b',
                r'\b(than|against|between)\b'
            ],
            QueryIntent.HOW_TO: [
                r'\b(how to|how do|how can|tutorial|guide|step)\b',
                r'\b(learn|teach|explain|show)\b'
            ],
            QueryIntent.DEFINITION: [
                r'\b(define|definition|meaning|what is|what are)\b',
                r'\b(explain|describe)\b'
            ],
            QueryIntent.NEWS: [
                r'\b(news|latest|recent|update|breaking|current)\b',
                r'\b(today|yesterday|this week|this month)\b'
            ],
            QueryIntent.OPINION: [
                r'\b(opinion|review|think|feel|believe|recommend)\b',
                r'\b(should|would|could|might)\b'
            ]
        }
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a query string.
        
        Args:
            query: The query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        query = query.strip()
        
        if len(query) < 3:
            return False, "Query must be at least 3 characters long"
        
        if len(query) > 500:
            return False, "Query must be less than 500 characters"
        
        # Check for suspicious patterns
        if re.search(r'[<>{}[\]\\]', query):
            return False, "Query contains invalid characters"
        
        # Check if query is mostly punctuation
        if len(re.sub(r'[^\w\s]', '', query)) < 2:
            return False, "Query must contain meaningful text"
        
        return True, None
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize a query string.
        
        Args:
            query: The raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove leading/trailing punctuation except question marks
        query = query.strip(string.punctuation.replace('?', ''))
        
        # Normalize case for common words
        query = re.sub(r'\b(AND|OR|NOT)\b', lambda m: m.group().lower(), query)
        
        return query
    
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the intent of a query.
        
        Args:
            query: The query string to classify
            
        Returns:
            Classified intent
        """
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score
        
        # Return the intent with the highest score, or OTHER if no matches
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return QueryIntent.OTHER
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query.
        
        Args:
            query: The query string
            
        Returns:
            List of extracted keywords
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def generate_search_variations(self, query: str) -> List[str]:
        """
        Generate search query variations for better results.
        
        Args:
            query: The original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        keywords = self.extract_keywords(query)
        
        if len(keywords) > 1:
            # Add keyword-only version
            variations.append(' '.join(keywords))
            
            # Add quoted version for exact phrases
            if len(query.split()) > 1:
                variations.append(f'"{query}"')
        
        return variations[:3]  # Limit to 3 variations