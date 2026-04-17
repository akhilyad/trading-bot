"""
NLP NEWS TRADING MODULE
Real-time news parsing, sentiment analysis, and automated trading on news events
"""

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import requests
from bs4 import BeautifulSoup
from logger import logger


class NewsFetcher:
    """Fetch news from multiple sources."""

    def __init__(self):
        self.news_cache = {}
        self.cache_duration = 300  # 5 minutes

    def fetch_moneycontrol(self, symbol: str) -> List[Dict]:
        """Fetch news from MoneyControl."""
        try:
            url = f"https://www.moneycontrol.com/stocks/cpsearch.php?search_str={symbol}&search_type=company"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            news_items = []
            articles = soup.find_all('a', href=True)[:10]

            for article in articles:
                title = article.text.strip()
                if title and len(title) > 10:
                    news_items.append({
                        'source': 'MoneyControl',
                        'title': title,
                        'url': article.get('href', ''),
                        'timestamp': datetime.now()
                    })

            return news_items

        except Exception as e:
            logger.error(f"MoneyControl fetch error: {e}")
            return []

    def fetch_screener(self, symbol: str) -> List[Dict]:
        """Fetch corporate actions from Screener."""
        try:
            url = f"https://www.screener.in/company/{symbol}/"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            news_items = []
            # Look for news section
            news_section = soup.find_all('li', class_='news-item')[:5]

            for item in news_section:
                title = item.text.strip()
                if title:
                    news_items.append({
                        'source': 'Screener.in',
                        'title': title,
                        'url': '',
                        'timestamp': datetime.now()
                    })

            return news_items

        except Exception as e:
            logger.error(f"Screener fetch error: {e}")
            return []

    def fetch_economic_times(self, symbol: str) -> List[Dict]:
        """Fetch from Economic Times."""
        try:
            url = f"https://economictimes.indiatimes.com/searchresult.cms?searchString={symbol}"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            news_items = []
            articles = soup.find_all('a', class_='news_t')[:10]

            for article in articles:
                title = article.text.strip()
                if title and len(title) > 10:
                    news_items.append({
                        'source': 'EconomicTimes',
                        'title': title,
                        'url': article.get('href', ''),
                        'timestamp': datetime.now()
                    })

            return news_items

        except Exception as e:
            logger.error(f"ET fetch error: {e}")
            return []

    def fetch_all_sources(self, symbol: str) -> List[Dict]:
        """Fetch news from all sources."""
        cache_key = f"{symbol}_{int(time.time() / self.cache_duration)}"

        if cache_key in self.news_cache:
            return self.news_cache[cache_key]

        all_news = []

        # Fetch from multiple sources
        all_news.extend(self.fetch_moneycontrol(symbol))
        all_news.extend(self.fetch_screener(symbol))
        all_news.extend(self.fetch_economic_times(symbol))

        # Deduplicate by title
        seen = set()
        unique_news = []
        for item in all_news:
            title = item['title'].lower()[:50]
            if title not in seen:
                seen.add(title)
                unique_news.append(item)

        self.news_cache[cache_key] = unique_news

        return unique_news


class NLPSentimentAnalyzer:
    """NLP-based sentiment analysis for news."""

    def __init__(self):
        # Sentiment keywords (simplified lexicon)
        self.positive_keywords = [
            'profit', 'growth', 'revenue', 'gain', 'surge', 'rally', 'bullish',
            'upgrade', 'buy', 'outperform', 'target', 'dividend', 'bonus',
            'record', 'high', 'strong', 'beat', 'exceed', 'positive', 'growth',
            'deal', 'contract', 'expansion', 'launch', 'innovation', 'partnership'
        ]

        self.negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'bearish', 'downgrade', 'sell',
            'underperform', 'cut', 'warning', 'risk', 'concern', 'slowdown',
            'weak', 'miss', 'disappoint', 'fraud', 'scandal', 'investigation',
            'fine', 'penalty', 'lawsuit', 'debt', 'default', 'bankruptcy'
        ]

        self.impact_keywords = {
            'high': ['results', 'earnings', 'quarter', 'fiscal', 'revenue', 'profit',
                    'acquisition', 'merger', 'deal', 'bonus', 'dividend', 'split'],
            'medium': ['appointment', 'expansion', 'launch', 'contract', 'order',
                      'partnership', 'funding', 'investment'],
            'low': ['statement', 'announcement', 'update', 'meeting', 'report']
        }

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of news text."""
        text_lower = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0
            sentiment_label = "NEUTRAL"
        else:
            sentiment_score = (positive_count - negative_count) / total

            if sentiment_score > 0.3:
                sentiment_label = "POSITIVE"
            elif sentiment_score < -0.3:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"

        # Determine impact
        impact = "low"
        for level, keywords in self.impact_keywords.items():
            if any(kw in text_lower for kw in keywords):
                impact = level
                break

        return {
            'sentiment': sentiment_label,
            'score': sentiment_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'impact': impact,
            'confidence': min(90, 50 + positive_count * 5 + negative_count * 5)
        }

    def analyze_news_batch(self, news_items: List[Dict]) -> Dict:
        """Analyze batch of news items."""
        if not news_items:
            return {
                'overall_sentiment': 'NEUTRAL',
                'score': 0,
                'impact': 'low',
                'news_count': 0,
                'top_news': []
            }

        sentiments = []
        for item in news_items:
            analysis = self.analyze_sentiment(item.get('title', ''))
            item['sentiment_analysis'] = analysis
            sentiments.append(analysis)

        # Aggregate
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)

        # Determine overall
        if avg_score > 0.2:
            overall = 'POSITIVE'
        elif avg_score < -0.2:
            overall = 'NEGATIVE'
        else:
            overall = 'NEUTRAL'

        # Highest impact news
        high_impact_news = [item for item in news_items
                          if item.get('sentiment_analysis', {}).get('impact') == 'high']

        return {
            'overall_sentiment': overall,
            'score': avg_score,
            'impact': max([s['impact'] for s in sentiments], key=['low', 'medium', 'high'].index),
            'confidence': avg_confidence,
            'news_count': len(news_items),
            'top_news': high_impact_news[:3]
        }


class NewsEventDetector:
    """Detect specific news events that trigger trading."""

    # Event patterns
    EVENT_PATTERNS = {
        'EARNINGS': r'(?:q[1-4]|quarter|results|earnings|revenue|profit).*(?:beat|miss|growth|decline)',
        'DIVIDEND': r'(?:dividend|bonus|share|subdivision|split)',
        'MERGER': r'(?:acquisition|merger|acqui|consolidation|buyout)',
        'GOVERNMENT': r'(?:government|regulatory|sebi|rbi|policy|licence)',
        'FUNDRAISING': r'(?:fund|raising|capital|issue|ipo|offer)',
        'LEGAL': r'(?:court|case|lawsuit|legal|investigation|police|fine|penalty)',
        'OPERATIONS': r'(?:plant|factory|shutdown|production|expansion|new product)'
    }

    def __init__(self):
        self.event_history = deque(maxlen=50)

    def detect_events(self, news_items: List[Dict]) -> List[Dict]:
        """Detect specific events in news."""
        detected_events = []

        for item in news_items:
            title = item.get('title', '').lower()

            for event_type, pattern in self.EVENT_PATTERNS.items():
                if re.search(pattern, title, re.IGNORECASE):
                    detected_events.append({
                        'type': event_type,
                        'title': item.get('title'),
                        'source': item.get('source'),
                        'timestamp': item.get('timestamp'),
                        'impact': item.get('sentiment_analysis', {}).get('impact', 'medium')
                    })
                    break

        # Record events
        for event in detected_events:
            self.event_history.append(event)

        return detected_events


class NewsTradingSignals:
    """Generate trading signals from news."""

    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = NLPSentimentAnalyzer()
        self.event_detector = NewsEventDetector()

    def get_signal(self, symbol: str) -> Dict:
        """Get trading signal based on news."""
        # Fetch latest news
        news = self.news_fetcher.fetch_all_sources(symbol)

        if not news:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'No news available',
                'sentiment': 'NEUTRAL',
                'impact': 'low'
            }

        # Analyze sentiment
        sentiment_analysis = self.sentiment_analyzer.analyze_news_batch(news)

        # Detect events
        events = self.event_detector.detect_events(news)

        # Generate signal based on sentiment and events
        action = "HOLD"
        confidence = 30
        reason = ""

        sentiment = sentiment_analysis['overall_sentiment']
        impact = sentiment_analysis['impact']
        score = sentiment_analysis['score']

        # High impact positive news = BUY
        if sentiment == "POSITIVE" and impact == "high" and score > 0.5:
            action = "BUY"
            confidence = min(85, 50 + score * 30 + sentiment_analysis['confidence'] * 0.3)
            reason = f"High-impact positive news: {news[0].get('title', '')[:50]}"

        # High impact negative news = SELL
        elif sentiment == "NEGATIVE" and impact == "high" and score < -0.5:
            action = "SELL"
            confidence = min(85, 50 + abs(score) * 30 + sentiment_analysis['confidence'] * 0.3)
            reason = f"High-impact negative news: {news[0].get('title', '')[:50]}"

        # Event-based signals
        for event in events:
            if event['type'] in ['EARNINGS', 'MERGER'] and event['impact'] == 'high':
                # Strong event - adjust confidence
                confidence += 15
                reason = f"Event: {event['type']} - {event['title'][:40]}"

        return {
            'action': action,
            'confidence': int(confidence),
            'reason': reason,
            'sentiment': sentiment,
            'impact': impact,
            'events': events,
            'news_count': len(news)
        }


class GoogleTrendsIntegration:
    """Integrate Google Trends data for retail sentiment."""

    def __init__(self):
        self.trends_cache = {}

    def get_trends_data(self, keyword: str) -> Optional[Dict]:
        """Get Google Trends data (simplified)."""
        # In production: use pytrends library
        # Simplified: return mock structure
        return {
            'keyword': keyword,
            'interest_over_time': [],
            'interest_by_region': [],
            'related_queries': []
        }

    def analyze_trend_signal(self, symbol: str) -> Dict:
        """Analyze Google trend for trading signal."""
        # This would use actual Google Trends API
        # For now: return neutral
        return {
            'signal': 'NEUTRAL',
            'interest_level': 'medium',
            'direction': 'stable'
        }


class SocialMediaSentiment:
    """Parse social media sentiment (Twitter, Reddit)."""

    def __init__(self):
        self.sentiment_cache = {}

    def get_sentiment(self, symbol: str) -> Dict:
        """Get social media sentiment (simplified)."""
        # In production: connect to Twitter API, Reddit API
        return {
            'twitter_sentiment': 'NEUTRAL',
            'reddit_sentiment': 'NEUTRAL',
            'overall': 'NEUTRAL',
            'buzz_level': 'low'
        }