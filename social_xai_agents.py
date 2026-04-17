"""
SOCIAL LISTENING MODULE
Twitter, Reddit, YouTube, Web Forums Sentiment Analysis
"""

import re
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from logger import logger


class TwitterScraper:
    """Twitter sentiment and trending analysis."""

    def __init__(self):
        self.tweets_cache = {}
        self.sentiment_history = defaultdict(list)

    def get_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from Twitter (simulated)."""
        # In production: Use Twitter API v2
        # For now: Simulated structure

        # Simulate key influencers
        influencers = [
            {'handle': '@elonmusk', 'sentiment': 'positive', 'influence': 95},
            {'handle': '@narendramodi', 'sentiment': 'positive', 'influence': 90},
            {'handle': '@WarrenBuffet', 'sentiment': 'neutral', 'influence': 85}
        ]

        # Simulate recent tweets about symbol
        tweets = [
            {'text': f'{symbol} shows strong momentum', 'sentiment': 'positive', 'likes': 150},
            {'text': f'Waiting for {symbol} earnings', 'sentiment': 'neutral', 'likes': 50},
            {'text': f'{symbol} breakout incoming?', 'sentiment': 'positive', 'likes': 200}
        ]

        # Calculate overall sentiment
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        for tweet in tweets:
            sentiment_scores[tweet['sentiment']] += 1

        total = sum(sentiment_scores.values())
        if total > 0:
            positive_pct = sentiment_scores['positive'] / total * 100
            negative_pct = sentiment_scores['negative'] / total * 100
            score = (positive_pct - negative_pct) / 100
        else:
            score = 0

        return {
            'overall_sentiment': 'POSITIVE' if score > 0.2 else 'NEGATIVE' if score < -0.2 else 'NEUTRAL',
            'score': score,
            'tweet_count': len(tweets),
            'influencers': influencers,
            'trending': score > 0.3,
            'buzz_level': 'HIGH' if len(tweets) > 5 else 'MEDIUM' if len(tweets) > 2 else 'LOW'
        }

    def get_trending_stocks(self) -> List[Dict]:
        """Get trending stock discussions."""
        # Simulated trending
        return [
            {'symbol': 'RELIANCE', 'mentions': 150, 'sentiment': 'positive'},
            {'symbol': 'TCS', 'mentions': 120, 'sentiment': 'neutral'},
            {'symbol': 'INFY', 'mentions': 100, 'sentiment': 'positive'}
        ]


class RedditScraper:
    """Reddit sentiment analysis."""

    def __init__(self):
        self.posts_cache = {}

    def get_subreddit_sentiment(self, subreddit: str) -> Dict:
        """Get sentiment from subreddit (simulated)."""
        # In production: Use Reddit API (PRAW)

        # Simulate posts
        posts = [
            {'title': 'Analysis: NIFTY Bullish setup', 'score': 250, 'comments': 45},
            {'title': 'What stocks are you watching?', 'score': 50, 'comments': 100},
            {'title': 'Bearish on IT sector', 'score': 80, 'comments': 30}
        ]

        # Calculate sentiment
        bullish = sum(1 for p in posts if 'bull' in p['title'].lower() or 'buy' in p['title'].lower())
        bearish = sum(1 for p in posts if 'bear' in p['title'].lower() or 'sell' in p['title'].lower())

        total = len(posts)
        score = (bullish - bearish) / total if total > 0 else 0

        return {
            'sentiment': 'BULLISH' if score > 0.3 else 'BEARISH' if score < -0.3 else 'NEUTRAL',
            'score': score,
            'post_count': len(posts),
            'total_comments': sum(p['comments'] for p in posts),
            'top_posts': posts[:3]
        }

    def get_wallstreetbets_style(self) -> Dict:
        """Get WSB-style sentiment (meme stocks, gamma squeeze potential)."""
        # Simulated
        return {
            'meme_stocks': ['RELIANCE', 'TATAMOTORS'],
            'gamma_squeeze_candidates': [],
            'sentiment': 'NEUTRAL',
            'mentions_24h': 500
        }


class YouTubeScraper:
    """YouTube video sentiment analysis."""

    def __init__(self):
        self.videos_cache = {}

    def get_chart_analysis_sentiment(self, symbol: str) -> Dict:
        """Analyze YouTube chart analysis videos."""
        # In production: YouTube Data API

        videos = [
            {'title': f'{symbol} Technical Analysis - Bullish!', 'views': 5000, 'likes': 200},
            {'title': f'{symbol} daily chart review', 'views': 3000, 'likes': 100},
            {'title': f'{symbol} support levels to watch', 'views': 2000, 'likes': 80}
        ]

        bullish_count = sum(1 for v in videos if 'bullish' in v['title'].lower() or 'buy' in v['title'].lower())
        bearish_count = sum(1 for v in videos if 'bearish' in v['title'].lower() or 'sell' in v['title'].lower())

        return {
            'sentiment': 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL',
            'video_count': len(videos),
            'total_views': sum(v['views'] for v in videos),
            'avg_engagement': sum(v['likes'] for v in videos) / len(videos) if videos else 0
        }


class WebForumMonitor:
    """Monitor various web forums."""

    FORUMS = [
        'TradingView',
        'StockTwits',
        'IndicatorForum',
        'Traderji'
    ]

    def __init__(self):
        self.forum_posts = defaultdict(list)

    def get_forum_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from multiple forums."""
        # Simulated data
        posts = [
            {'forum': 'TradingView', 'sentiment': 'positive', 'replies': 25},
            {'forum': 'StockTwits', 'sentiment': 'neutral', 'replies': 15},
            {'forum': 'Traderji', 'sentiment': 'negative', 'replies': 10}
        ]

        sentiment_counts = defaultdict(int)
        total_replies = 0

        for post in posts:
            sentiment_counts[post['sentiment']] += 1
            total_replies += post['replies']

        return {
            'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get),
            'forum_count': len(posts),
            'total_activity': total_replies,
            'posts': posts
        }


class SocialListeningEngine:
    """Complete social listening system."""

    def __init__(self):
        self.twitter = TwitterScraper()
        self.reddit = RedditScraper()
        self.youtube = YouTubeScraper()
        self.forums = WebForumMonitor()

        self.symbol_sentiment = defaultdict(lambda: {
            'twitter': {},
            'reddit': {},
            'youtube': {},
            'forums': {},
            'combined_score': 0
        })

    def update_symbol(self, symbol: str):
        """Update all social data for a symbol."""
        # Get from all sources
        twitter_sent = self.twitter.get_sentiment(symbol)
        reddit_sent = self.reddit.get_subreddit_sentiment('IndiaInvestments')
        youtube_sent = self.youtube.get_chart_analysis_sentiment(symbol)
        forum_sent = self.forums.get_forum_sentiment(symbol)

        # Store
        self.symbol_sentiment[symbol] = {
            'twitter': twitter_sent,
            'reddit': reddit_sent,
            'youtube': youtube_sent,
            'forums': forum_sent,
            'last_updated': datetime.now()
        }

        # Calculate combined score
        scores = []
        if twitter_sent.get('score'):
            scores.append(twitter_sent['score'])
        if reddit_sent.get('score'):
            scores.append(reddit_sent['score'])

        self.symbol_sentiment[symbol]['combined_score'] = sum(scores) / len(scores) if scores else 0

    def get_sentiment(self, symbol: str) -> Dict:
        """Get combined sentiment for symbol."""
        if symbol not in self.symbol_sentiment:
            self.update_symbol(symbol)

        data = self.symbol_sentiment[symbol]

        # Generate trading signal
        score = data['combined_score']
        twitter_trend = data['twitter'].get('trending', False)
        reddit_momentum = data['reddit'].get('sentiment') == 'BULLISH'
        youtube_bullish = data['youtube'].get('sentiment') == 'BULLISH'

        signals = []
        if score > 0.3:
            signals.append('BUY')
        elif score < -0.3:
            signals.append('SELL')

        if twitter_trend and score > 0:
            signals.append('STRONG_BUY')

        if reddit_momentum:
            signals.append('REDDIT_MOMENTUM')

        # Combined signal
        action = 'HOLD'
        confidence = 40

        if 'STRONG_BUY' in signals:
            action = 'BUY'
            confidence = 80
        elif 'BUY' in signals and 'SELL' not in signals:
            action = 'BUY'
            confidence = 65
        elif 'SELL' in signals and 'BUY' not in signals:
            action = 'SELL'
            confidence = 65

        return {
            'action': action,
            'confidence': confidence,
            'sources': {
                'twitter': data['twitter'],
                'reddit': data['reddit'],
                'youtube': data['youtube'],
                'forums': data['forums']
            },
            'combined_score': score,
            'signal_count': len(signals),
            'last_updated': data.get('last_updated')
        }

    def get_multi_symbol_comparison(self, symbols: List[str]) -> List[Dict]:
        """Compare sentiment across multiple symbols."""
        results = []

        for symbol in symbols:
            self.update_symbol(symbol)
            sent = self.get_sentiment(symbol)

            results.append({
                'symbol': symbol,
                'action': sent['action'],
                'confidence': sent['confidence'],
                'score': sent['combined_score']
            })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results


# =============================================================================
# EXPLAINABLE AI (XAI) MODULE
# =============================================================================

class SHAPExplainer:
    """SHAP values for model interpretability."""

    def __init__(self):
        self.feature_importance = {}

    def calculate_shap(self, features: Dict, prediction: str) -> Dict:
        """Calculate SHAP values for prediction."""
        # Simplified SHAP (in production: use shap library)

        # Assign importance based on feature values
        importance = {}

        feature_values = {
            'rsi': features.get('rsi', 50),
            'momentum': features.get('change_5d', 0),
            'volume': features.get('volume_ratio', 1),
            'volatility': features.get('atr_percent', 2),
            'trend': features.get('trend_strength', 0)
        }

        # Calculate contribution
        for feature, value in feature_values.items():
            if feature == 'rsi':
                contribution = -abs(value - 50) / 50  # Extreme RSI = important
            elif feature == 'momentum':
                contribution = abs(value) / 10  # Strong momentum = important
            elif feature == 'volume':
                contribution = (value - 1) * 0.5  # High volume = important
            elif feature == 'volatility':
                contribution = value / 5  # High volatility = important
            else:
                contribution = abs(value) / 10

            importance[feature] = contribution

        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'prediction': prediction,
            'base_value': 0.5,
            'shap_values': dict(sorted_importance),
            'top_features': [f[0] for f in sorted_importance[:3]]
        }

    def explain_decision(self, features: Dict, prediction: str) -> str:
        """Generate human-readable explanation."""
        shap = self.calculate_shap(features, prediction)

        explanation = f"Prediction: {prediction}\n"
        explanation += f"Reasoning (top features):\n"

        for feature in shap['top_features']:
            value = features.get(feature, 0)
            contribution = shap['shap_values'].get(feature, 0)

            if contribution > 0:
                explanation += f"  - {feature}={value:.2f} pushes towards BUY\n"
            else:
                explanation += f"  - {feature}={value:.2f} pushes towards SELL\n"

        return explanation


class CounterfactualExplainer:
    """Generate counterfactual explanations."""

    def __init__(self):
        pass

    def generate(self, features: Dict, prediction: str) -> List[Dict]:
        """Generate counterfactual scenarios."""
        counterfactuals = []

        # What needs to change for opposite prediction
        if prediction == 'BUY':
            # What would make it SELL?
            counterfactuals.append({
                'change': 'RSI above 70',
                'feature': 'rsi',
                'value': 75,
                'would_result': 'SELL'
            })
            counterfactuals.append({
                'change': 'Price below 20-day MA',
                'feature': 'ma_20',
                'value': features.get('close', 0) * 0.95,
                'would_result': 'SELL'
            })
        elif prediction == 'SELL':
            counterfactuals.append({
                'change': 'RSI below 30',
                'feature': 'rsi',
                'value': 25,
                'would_result': 'BUY'
            })

        return counterfactuals


class ExplainableAI:
    """Complete XAI system."""

    def __init__(self):
        self.shap = SHAPExplainer()
        self.counterfactual = CounterfactualExplainer()

    def explain(self, features: Dict, prediction: str) -> Dict:
        """Generate complete explanation."""
        return {
            'shap_values': self.shap.calculate_shap(features, prediction),
            'counterfactuals': self.counterfactual.generate(features, prediction),
            'human_readable': self.shap.explain_decision(features, prediction)
        }


# =============================================================================
# MULTI-AGENT COORDINATION SYSTEM
# =============================================================================

class Agent:
    """Base class for trading agent."""

    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.strategy_type = strategy_type
        self.performance = {'trades': 0, 'pnl': 0, 'win_rate': 0}

    def analyze(self, market_data: Dict) -> Dict:
        """Analyze and return signal."""
        return {'action': 'HOLD', 'confidence': 0}

    def update_performance(self, pnl: float):
        """Update agent performance."""
        self.performance['trades'] += 1
        self.performance['pnl'] += pnl
        self.performance['win_rate'] = (self.performance['win_rate'] * (self.performance['trades'] - 1) + (1 if pnl > 0 else 0)) / self.performance['trades']


class MomentumAgent(Agent):
    """Agent specialized in trend/momentum strategies."""

    def __init__(self):
        super().__init__("Momentum Agent", "MOMENTUM")

    def analyze(self, market_data: Dict) -> Dict:
        indicators = market_data.get('indicators', {})

        trend = indicators.get('trend_strength', 0)
        rsi = indicators.get('rsi', 50)

        if trend > 2 and rsi < 70:
            return {'action': 'BUY', 'confidence': 75, 'reason': 'Strong momentum'}
        elif trend < -2 and rsi > 30:
            return {'action': 'SELL', 'confidence': 75, 'reason': 'Bearish momentum'}

        return {'action': 'HOLD', 'confidence': 40}


class MeanReversionAgent(Agent):
    """Agent specialized in mean reversion."""

    def __init__(self):
        super().__init__("Mean Reversion Agent", "MEAN_REVERSION")

    def analyze(self, market_data: Dict) -> Dict:
        indicators = market_data.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        bb_pos = indicators.get('bb_position', 0.5)

        if rsi < 35 or bb_pos < 0.2:
            return {'action': 'BUY', 'confidence': 70, 'reason': 'Oversold'}
        elif rsi > 65 or bb_pos > 0.8:
            return {'action': 'SELL', 'confidence': 70, 'reason': 'Overbought'}

        return {'action': 'HOLD', 'confidence': 40}


class VolatilityAgent(Agent):
    """Agent specialized in volatility events."""

    def __init__(self):
        super().__init__("Volatility Agent", "VOLATILITY")

    def analyze(self, market_data: Dict) -> Dict:
        indicators = market_data.get('indicators', {})
        atr = indicators.get('atr_percent', 2)
        vol_regime = indicators.get('volatility_regime', 'NORMAL')

        if vol_regime == 'HIGH_VOL' and atr > 4:
            return {'action': 'HOLD', 'confidence': 60, 'reason': 'Wait for volatility to normalize'}

        return {'action': 'HOLD', 'confidence': 30}


class RiskAgent(Agent):
    """Agent specialized in risk management."""

    def __init__(self):
        super().__init__("Risk Agent", "RISK")

    def analyze(self, market_data: Dict) -> Dict:
        portfolio = market_data.get('portfolio', {})
        var = portfolio.get('var_99', 0)
        exposure = portfolio.get('exposure', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)

        # Risk checks
        if var > 5:
            return {'action': 'REDUCE_EXPOSURE', 'confidence': 90, 'reason': f'VaR too high: {var}%'}

        if exposure > 0.7:
            return {'action': 'CLOSE_POSITIONS', 'confidence': 80, 'reason': 'Overexposed'}

        if daily_pnl < -5000:
            return {'action': 'STOP_TRADING', 'confidence': 95, 'reason': 'Daily loss limit'}

        return {'action': 'HOLD', 'confidence': 50}


class CoordinatorAgent:
    """Coordinator agent that allocates capital between agents."""

    def __init__(self):
        self.agents = {
            'momentum': MomentumAgent(),
            'mean_reversion': MeanReversionAgent(),
            'volatility': VolatilityAgent(),
            'risk': RiskAgent()
        }
        self.allocation = {
            'momentum': 0.30,
            'mean_reversion': 0.30,
            'volatility': 0.20,
            'risk': 0.20
        }

    def coordinate(self, market_data: Dict) -> Dict:
        """Coordinate all agents and allocate capital."""
        # Get signals from each agent
        signals = {}
        for name, agent in self.agents.items():
            signals[name] = agent.analyze(market_data)

        # Risk agent has veto power
        if signals['risk']['action'] in ['STOP_TRADING', 'REDUCE_EXPOSURE']:
            return {
                'action': signals['risk']['action'],
                'confidence': signals['risk']['confidence'],
                'reason': signals['risk']['reason'],
                'by_agent': 'risk'
            }

        # Weight by allocation and confidence
        weighted_signals = {}
        for name, signal in signals.items():
            if signal['action'] == 'HOLD':
                continue

            weight = self.allocation[name]
            score = signal['confidence'] * weight
            weighted_signals[name] = score

        if not weighted_signals:
            return {'action': 'HOLD', 'confidence': 40}

        # Select best signal
        best_agent = max(weighted_signals, key=weighted_signals.get)
        best_signal = signals[best_agent]

        return {
            'action': best_signal['action'],
            'confidence': best_signal['confidence'],
            'reason': f"{best_signal.get('reason', '')} ({best_agent})",
            'by_agent': best_agent,
            'agent_votes': signals
        }

    def adjust_allocation(self, performance: Dict):
        """Adjust allocation based on performance."""
        total_pnl = sum(p.get('pnl', 0) for p in performance.values())

        if total_pnl <= 0:
            # Reduce exposure, increase risk management
            self.allocation['risk'] = 0.3
            self.allocation['momentum'] = 0.25
            self.allocation['mean_reversion'] = 0.25
            self.allocation['volatility'] = 0.2
        else:
            # Normal allocation
            pass