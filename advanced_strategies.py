"""
OPTIONS STRATEGY MODULE
Options-based trading: Hedging, Income, Volatility plays
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from logger import logger


class OptionsStrategy:
    """Options trading strategies for NSE/BSE."""

    def __init__(self):
        self.options_cache = {}

    def calculate_greeks(self, spot_price: float, strike_price: float,
                         time_to_expiry: float, volatility: float,
                         risk_free_rate: float = 0.07) -> Dict:
        """Calculate option Greeks using Black-Scholes."""
        # Time in years
        T = max(time_to_expiry / 365, 0.001)

        # d1 and d2
        d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        # Call Greeks
        call_price = spot_price * self._cdf(d1) - strike_price * np.exp(-risk_free_rate * T) * self._cdf(d2)
        call_delta = self._cdf(d1)
        call_gamma = (np.exp(-0.5 * d1**2) / (spot_price * volatility * np.sqrt(2 * np.pi * T)))
        call_vega = (spot_price * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi))
        call_theta = (-(spot_price * volatility * np.exp(-0.5 * d1**2) / (2 * np.sqrt(2 * np.pi * T))) -
                      risk_free_rate * strike_price * np.exp(-risk_free_rate * T) * self._cdf(d2))

        # Put Greeks
        put_price = strike_price * np.exp(-risk_free_rate * T) * self._cdf(-d2) - spot_price * self._cdf(-d1)
        put_delta = call_delta - 1
        put_gamma = call_gamma
        put_vega = call_vega
        put_theta = call_theta + risk_free_rate * strike_price * np.exp(-risk_free_rate * T) * self._cdf(-d2)

        return {
            'call_price': call_price,
            'put_price': put_price,
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': call_gamma,
            'vega': call_vega,
            'call_theta': call_theta,
            'put_theta': put_theta
        }

    def _cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def get_options_chain(self, symbol: str, expiry_days: int = 30) -> Dict:
        """Get simulated options chain for a symbol."""
        # In production, fetch from broker API
        # Here: generate based on spot + strike ladder

        # This would connect to Kite's options chain API
        # For now, return structure
        return {
            'symbol': symbol,
            'expiry': (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d'),
            'calls': [],
            'puts': []
        }

    def protective_put(self, symbol: str, entry_price: float,
                      quantity: int, spot_price: float,
                      days_to_expiry: int = 30) -> Optional[Dict]:
        """Buy protective put - hedge long position."""
        # Buy put at 5% below spot
        strike = spot_price * 0.95
        greeks = self.calculate_greeks(spot_price, strike, days_to_expiry, 0.25)

        # Cost of put (premium)
        premium = greeks['put_price']

        # Max loss = strike - entry + premium
        max_loss = (strike - entry_price) + premium if entry_price > strike else premium

        return {
            'strategy': 'PROTECTIVE_PUT',
            'action': 'BUY_PUT',
            'strike': strike,
            'premium': premium,
            'max_loss': max_loss,
            'reason': f"Hedge: Buy put at {strike} to protect against downside"
        }

    def covered_call(self, symbol: str, entry_price: float,
                     quantity: int, spot_price: float,
                     days_to_expiry: int = 30) -> Optional[Dict]:
        """Sell covered call - income generation."""
        # Sell call 5% above spot
        strike = spot_price * 1.05
        greeks = self.calculate_greeks(spot_price, strike, days_to_expiry, 0.25)

        premium = greeks['call_price']

        # Max profit = (strike - entry) + premium (if called away)
        max_profit = (strike - entry_price) + premium

        return {
            'strategy': 'COVERED_CALL',
            'action': 'SELL_CALL',
            'strike': strike,
            'premium': premium,
            'max_profit': max_profit,
            'reason': f"Income: Sell call at {strike} for premium"
        }

    def straddle(self, symbol: str, spot_price: float,
                 days_to_expiry: int = 30, volatility: float = 0.25) -> Optional[Dict]:
        """Buy straddle - volatility play."""
        # ATM strike
        strike = round(spot_price / 10) * 10  # Round to nearest 10
        greeks = self.calculate_greeks(spot_price, strike, days_to_expiry, volatility)

        total_premium = greeks['call_price'] + greeks['put_price']

        # Profit if move > premium paid
        breakeven = total_premium

        return {
            'strategy': 'STRADDLE',
            'action': 'BUY_STRADDLE',
            'strike': strike,
            'premium': total_premium,
            'break_even': break_even,
            'reason': f"Volatility: Buy call+put at {strike}, profit if move > ₹{break_even:.2f}"
        }

    def iron_condor(self, symbol: str, spot_price: float,
                   days_to_expiry: int = 30) -> Optional[Dict]:
        """Iron condor - range-bound income."""
        # Sell OTM call and put, buy further OTM for protection
        call_strike = spot_price * 1.05
        put_strike = spot_price * 0.95
        call_buy_strike = spot_price * 1.10
        put_buy_strike = spot_price * 0.90

        # Simplified premium calculation
        greeks_call = self.calculate_greeks(spot_price, call_strike, days_to_expiry, 0.25)
        greeks_put = self.calculate_greeks(spot_price, put_strike, days_to_expiry, 0.25)
        greeks_call_buy = self.calculate_greeks(spot_price, call_buy_strike, days_to_expiry, 0.25)
        greeks_put_buy = self.calculate_greeks(spot_price, put_buy_strike, days_to_expiry, 0.25)

        net_credit = (greeks_call['call_price'] + greeks_put['put_price'] -
                     greeks_call_buy['call_price'] - greeks_put_buy['put_price'])

        return {
            'strategy': 'IRON_CONDOR',
            'action': 'SELL_IRON_CONDOR',
            'call_strike': call_strike,
            'put_strike': put_strike,
            'net_credit': net_credit,
            'max_profit': net_credit,
            'max_loss': (call_strike - put_strike) / 2 - net_credit,
            'reason': f"Range: Iron condor for ₹{net_credit:.2f} credit"
        }

    def should_hedge(self, positions: Dict, portfolio_value: float,
                     market_regime: str) -> Optional[Dict]:
        """Determine if hedging is needed."""
        # Calculate portfolio delta
        total_delta = 0
        for symbol, pos in positions.items():
            delta = pos.quantity * pos.entry_price
            if pos.type == "LONG":
                total_delta += delta
            else:
                total_delta -= delta

        # Hedge if:
        # 1. Bear market + large long exposure
        # 2. High volatility expected
        # 3. Large portfolio value at risk

        exposure_ratio = abs(total_delta) / portfolio_value if portfolio_value > 0 else 0

        if "BEAR" in market_regime and exposure_ratio > 0.5:
            # Buy puts to hedge
            return {
                'hedge_type': 'PORTFOLIO_PUT',
                'action': 'BUY_PUTS',
                'protection': 0.2,  # 20% downside protection
                'reason': 'Bear market - protecting long portfolio'
            }

        if "HIGHVOL" in market_regime and exposure_ratio > 0.7:
            return {
                'hedge_type': 'COLLAR',
                'action': 'COLLAR',
                'protection': 0.15,
                'reason': 'High volatility - reducing exposure'
            }

        return None


# =============================================================================
# NEWS & SENTIMENT MODULE
# =============================================================================

class NewsSentiment:
    """News analysis and sentiment scoring."""

    def __init__(self):
        self.news_cache = {}
        self.sentiment_history = deque(maxlen=100)

    def fetch_news(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Fetch news for symbols (simulated)."""
        # In production: scrape screener.in, moneycontrol, economic times
        # For now: return simulated structure
        news = {}

        for symbol in symbols:
            # Simulated news data
            news[symbol] = [
                {
                    'title': f'{symbol} announces quarterly results',
                    'sentiment': 'positive',
                    'impact': 'high',
                    'timestamp': datetime.now()
                }
            ]

        return news

    def analyze_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze overall sentiment from news."""
        if not news_items:
            return {'sentiment': 'neutral', 'score': 0, 'news_count': 0}

        sentiment_scores = {
            'very_positive': 1.0,
            'positive': 0.5,
            'neutral': 0,
            'negative': -0.5,
            'very_negative': -1.0
        }

        impact_weights = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }

        total_score = 0
        total_weight = 0

        for news in news_items:
            sentiment = news.get('sentiment', 'neutral')
            impact = news.get('impact', 'medium')

            score = sentiment_scores.get(sentiment, 0)
            weight = impact_weights.get(impact, 1.0)

            total_score += score * weight
            total_weight += weight

        avg_score = total_score / total_weight if total_weight > 0 else 0

        # Categorize
        if avg_score > 0.5:
            sentiment = 'very_positive'
        elif avg_score > 0.2:
            sentiment = 'positive'
        elif avg_score > -0.2:
            sentiment = 'neutral'
        elif avg_score > -0.5:
            sentiment = 'negative'
        else:
            sentiment = 'very_negative'

        return {
            'sentiment': sentiment,
            'score': avg_score,
            'news_count': len(news_items)
        }

    def should_trade_news(self, symbol: str, regime: str) -> bool:
        """Determine if should trade around news."""
        # Don't trade during high-impact news events in certain regimes
        if "SIDEWAYS" in regime:
            return False  # News might break range

        if "HIGHVOL" in regime:
            return False  # Already volatile

        return True

    def get_event_calendar(self) -> List[Dict]:
        """Get upcoming events that might impact markets."""
        # In production: fetch from economic calendar
        events = [
            {'event': 'RBI Policy Meeting', 'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'), 'impact': 'high'},
            {'event': 'FII Data', 'date': datetime.now().strftime('%Y-%m-%d'), 'impact': 'medium'},
            {'event': 'NIFTY Expiry', 'date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'), 'impact': 'medium'},
        ]
        return events


# =============================================================================
# MARKET DEPTH & ORDER FLOW
# =============================================================================

class MarketDepth:
    """Order flow and market depth analysis."""

    def __init__(self):
        self.depth_cache = {}

    def get_ltp(self, symbol: str) -> Optional[Dict]:
        """Get Last Traded Price with volume analysis."""
        # In production: connect to broker's quote API
        return {
            'price': 0,
            'volume': 0,
            'vwap': 0,
            'change': 0
        }

    def calculate_buy_sell_pressure(self, trades: List[Dict]) -> Dict:
        """Calculate buy/sell pressure from trade data."""
        buy_volume = 0
        sell_volume = 0

        for trade in trades:
            if trade.get('side') == 'buy':
                buy_volume += trade.get('volume', 0)
            else:
                sell_volume += trade.get('volume', 0)

        total = buy_volume + sell_volume

        if total == 0:
            return {'pressure': 'neutral', 'ratio': 1.0}

        ratio = buy_volume / total

        if ratio > 0.6:
            pressure = 'strong_buy'
        elif ratio > 0.55:
            pressure = 'buy'
        elif ratio < 0.4:
            pressure = 'strong_sell'
        elif ratio < 0.45:
            pressure = 'sell'
        else:
            pressure = 'neutral'

        return {
            'pressure': pressure,
            'ratio': ratio,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume
        }

    def detect_large_trades(self, trades: List[Dict], threshold_percent: float = 0.1) -> List[Dict]:
        """Detect large institutional trades."""
        avg_volume = sum(t.get('volume', 0) for t in trades) / len(trades) if trades else 0

        large_trades = []
        for trade in trades:
            if trade.get('volume', 0) > avg_volume * (1 + threshold_percent):
                large_trades.append(trade)

        return large_trades

    def volume_profile(self, prices: List[float], volumes: List[float]) -> Dict:
        """Calculate Volume Profile - Point of Control."""
        if not prices or not volumes:
            return {'poc': 0, 'va': (0, 0)}

        # Group by price bins
        price_bins = {}
        for price, volume in zip(prices, volumes):
            bin_price = int(price / 10) * 10  # Bin size 10
            price_bins[bin_price] = price_bins.get(bin_price, 0) + volume

        # Point of Control (highest volume)
        poc = max(price_bins, key=price_bins.get)

        # Value Area (70% of volume)
        sorted_bins = sorted(price_bins.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(price_bins.values())
        cumsum = 0
        va_low = poc
        va_high = poc

        for price, vol in sorted_bins:
            cumsum += vol
            if cumsum / total_volume <= 0.35:
                if price < poc:
                    va_low = price
                elif price > poc:
                    va_high = price

        return {
            'poc': poc,
            'va_low': va_low,
            'va_high': va_high,
            'profile': price_bins
        }


# =============================================================================
# SECTOR ROTATION MODULE
# =============================================================================

class SectorRotation:
    """Sector analysis and rotation strategy."""

    # NSE sector mapping
    SECTORS = {
        'RELIANCE': 'ENERGY',
        'TCS': 'IT',
        'INFY': 'IT',
        'HDFCBANK': 'FINANCIAL',
        'ICICIBANK': 'FINANCIAL',
        'SBIN': 'FINANCIAL',
        'KOTAKBANK': 'FINANCIAL',
        'ITC': 'FMCG',
        'HINDUNILVR': 'FMCG',
        'MARUTI': 'AUTO',
        'TATAMOTORS': 'AUTO',
        'M&M': 'AUTO',
        'BHARTIARTL': 'TELECOM',
        'WIPRO': 'IT',
        'HCLTECH': 'IT',
    }

    def __init__(self):
        self.sector_performance = defaultdict(list)

    def get_sector_performance(self, symbols: List[str],
                              prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance by sector."""
        sector_returns = defaultdict(list)

        for symbol, price in prices.items():
            sector = self.SECTORS.get(symbol, 'OTHER')

            # Calculate return (simplified - would use actual data)
            # In production: compare to previous close
            ret = random.uniform(-2, 2)  # Simulated
            sector_returns[sector].append(ret)

        # Average return per sector
        sector_perf = {}
        for sector, returns in sector_returns.items():
            sector_perf[sector] = sum(returns) / len(returns)

        return sector_perf

    def get_leading_sectors(self, sector_perf: Dict) -> List[str]:
        """Get top performing sectors."""
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sectors[:3]]

    def get_lagging_sectors(self, sector_perf: Dict) -> List[str]:
        """Get worst performing sectors."""
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1])
        return [s[0] for s in sorted_sectors[:2]]

    def should_rotate(self, regime: str, sector_perf: Dict) -> bool:
        """Determine if sector rotation is needed."""
        leading = self.get_leading_sectors(sector_perf)
        lagging = self.get_lagging_sectors(sector_perf)

        if not leading or not lagging:
            return False

        spread = sector_perf[leading[0]] - sector_perf[lagging[0]]

        # Rotate if spread > 3%
        return spread > 3

    def get_rotation_signal(self, regime: str) -> str:
        """Get sector rotation recommendation based on regime."""
        rotation_map = {
            'BULL': 'LEADERS',  # Stay with leaders
            'BEAR': 'DEFENSIVE',  # Move to FMCG, IT
            'SIDEWAYS': 'VALUE',  # Value stocks
            'FII_BUYING': 'LEADERS',
            'FII_SELLING': 'DEFENSIVE'
        }

        return rotation_map.get(regime, 'BALANCED')


# =============================================================================
# VOLATILITY FORECASTING
# =============================================================================

class VolatilityForecast:
    """Predict and trade volatility."""

    def __init__(self):
        self.volatility_history = deque(maxlen=100)

    def calculate_current_volatility(self, returns: List[float]) -> float:
        """Calculate realized volatility."""
        if len(returns) < 2:
            return 0

        return np.std(returns) * np.sqrt(252)  # Annualized

    def predict_volatility(self, regime: str, current_vol: float,
                          atr_percent: float) -> str:
        """Predict future volatility regime."""
        self.volatility_history.append(current_vol)

        avg_vol = np.mean(list(self.volatility_history)) if self.volatility_history else current_vol

        # Volatility regime
        if current_vol > avg_vol * 1.5:
            return 'HIGH_VOL'
        elif current_vol < avg_vol * 0.7:
            return 'LOW_VOL'
        else:
            return 'NORMAL_VOL'

    def get_volatility_adjusted_stops(self, entry: float, atr: float,
                                     regime: str) -> Tuple[float, float]:
        """Get volatility-adjusted stops."""
        if regime == 'HIGH_VOL':
            # Wider stops
            stop = entry - (2.5 * atr)
            target = entry + (5 * atr)  # 2:1 R:R
        elif regime == 'LOW_VOL':
            # Tighter stops
            stop = entry - (1.5 * atr)
            target = entry + (3 * atr)
        else:
            # Normal
            stop = entry - (2 * atr)
            target = entry + (4 * atr)

        return stop, target

    def should_use_options(self, volatility_regime: str, regime: str) -> bool:
        """Determine if options strategies are better."""
        # High volatility = use options for cost reduction
        if volatility_regime == 'HIGH_VOL':
            return True

        # Earnings/events = options
        # In production: check event calendar

        return False


# =============================================================================
# ADVANCED TECHNICAL SYSTEMS
# =============================================================================

class AdvancedTechnical:
    """Ichimoku, Volume Profile, Fibonacci, Harmonics."""

    def ichimoku_cloud(self, high: pd.Series, low: pd.Series,
                      close: pd.Series) -> Dict:
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2

        # Kijun-sen (Base Line)
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-26)

        # Current values
        current_tenkan = tenkan.iloc[-1]
        current_kijun = kijun.iloc[-1]
        current_senkou_a = senkou_a.iloc[-1]
        current_senkou_b = senkou_b.iloc[-1]
        current_price = close.iloc[-1]

        # Bullish signal: price above cloud, tenkan > kijun
        bullish = (current_price > current_senkou_a and
                  current_price > current_senkou_b and
                  current_tenkan > current_kijun)

        # Bearish signal: price below cloud, tenkan < kijun
        bearish = (current_price < current_senkou_a and
                  current_price < current_senkou_b and
                  current_tenkan < current_kijun)

        return {
            'tenkan': current_tenkan,
            'kijun': current_kijun,
            'senkou_a': current_senkou_a,
            'senkou_b': current_senkou_b,
            'cloud_top': max(current_senkou_a, current_senkou_b),
            'cloud_bottom': min(current_senkou_a, current_senkou_b),
            'signal': 'BUY' if bullish else 'SELL' if bearish else 'NEUTRAL'
        }

    def fibonacci_retracements(self, high: float, low: float) -> Dict:
        """Calculate Fibonacci retracement levels."""
        diff = high - low

        levels = {
            '0%': high,
            '23.6%': high - (0.236 * diff),
            '38.2%': high - (0.382 * diff),
            '50%': high - (0.5 * diff),
            '61.8%': high - (0.618 * diff),
            '78.6%': high - (0.786 * diff),
            '100%': low
        }

        return levels

    def fibonacci_extensions(self, swing_low: float, swing_high: float,
                           pullback_low: float) -> Dict:
        """Calculate Fibonacci extension levels for targets."""
        diff = swing_high - swing_low

        # Extension from pullback low
        levels = {
            '127.2%': pullback_low + (1.272 * diff),
            '161.8%': pullback_low + (1.618 * diff),
            '200%': pullback_low + (2.0 * diff),
            '261.8%': pullback_low + (2.618 * diff)
        }

        return levels

    def detect_harmonic_pattern(self, prices: pd.Series) -> Optional[Dict]:
        """Detect basic harmonic patterns (simplified)."""
        # This is complex - would need zigzag analysis
        # Return structure for pattern detection

        return {
            'pattern': None,
            'signal': 'NEUTRAL',
            'reversal_point': None
        }

    def volume_profile_analysis(self, df: pd.DataFrame) -> Dict:
        """Volume Profile analysis."""
        if 'volume' not in df.columns:
            return {}

        # Group by price level
        df_copy = df.copy()
        df_copy['price_bin'] = (df_copy['close'] / 10).astype(int) * 10

        volume_profile = df_copy.groupby('price_bin')['volume'].sum()

        # Point of Control
        poc_price = volume_profile.idxmax()
        poc_volume = volume_profile.max()

        # Value Area (70%)
        total_volume = volume_profile.sum()
        cumsum = 0
        va_low = poc_price
        va_high = poc_price

        for price, vol in volume_profile.sort_values(ascending=False).items():
            cumsum += vol
            if cumsum / total_volume <= 0.35:
                if price < poc_price:
                    va_low = price
                elif price > poc_price:
                    va_high = price
            else:
                break

        # Current price position
        current_price = df['close'].iloc[-1]

        if current_price > va_high:
            position = 'ABOVE_VA'  # Bullish
        elif current_price < va_low:
            position = 'BELOW_VA'  # Bearish
        else:
            position = 'IN_VA'  # Neutral

        return {
            'poc': poc_price,
            'poc_volume': poc_volume,
            'va_low': va_low,
            'va_high': va_high,
            'current_position': position,
            'volume_profile': volume_profile.to_dict()
        }


# =============================================================================
# MARKET BREADTH INDICATORS
# =============================================================================

class MarketBreadth:
    """Market breadth and macro indicators."""

    def __init__(self):
        self.breadth_history = deque(maxlen=100)

    def calculate_adn(self, advances: int, declines: int) -> float:
        """Calculate Advance/Decline ratio."""
        total = advances + declines
        if total == 0:
            return 0

        return (advances / total) * 100

    def get_breadth_signal(self, nifty_change: float, adn: float,
                          new_highs: int, new_lows: int) -> str:
        """Generate breadth confirmation signal."""
        # Strong bullish: NIFTY up + high ANDN + new highs > new lows
        if nifty_change > 0 and adn > 55 and new_highs > new_lows:
            return 'STRONG_BULLISH'

        # Bullish: NIFTY up + moderate ADN
        if nifty_change > 0 and adn > 50:
            return 'BULLISH'

        # Strong bearish
        if nifty_change < 0 and adn < 45 and new_lows > new_highs:
            return 'STRONG_BEARISH'

        # Bearish
        if nifty_change < 0 and adn < 50:
            return 'BEARISH'

        return 'NEUTRAL'

    def fii_flow_signal(self, fii_buy: float, dii_buy: float) -> str:
        """FII/DII flow analysis."""
        if fii_buy > dii_buy * 1.5 and fii_buy > 1000:
            return 'FII_DOMINANT_BUY'
        elif fii_buy < -1000 and dii_buy > 0:
            return 'DII_SUPPORT'
        elif fii_buy < -dii_buy * 1.5:
            return 'FII_DOMINANT_SELL'

        return 'BALANCED'

    def currency_correlation(self, usd_inr: float, nifty: float,
                           prev_usd_inr: float, prev_nifty: float) -> str:
        """Analyze USD/INR correlation with NIFTY."""
        usd_change = (usd_inr - prev_usd_inr) / prev_usd_inr * 100
        nifty_change = (nifty - prev_nifty) / prev_nifty * 100

        # Negative correlation typically
        if usd_change > 0.5 and nifty_change < -0.5:
            return 'CURRENCY_HEADWIND'
        elif usd_change < -0.5 and nifty_change > 0.5:
            return 'CURRENCY_TAILWIND'

        return 'CURRENCY_NEUTRAL'


# =============================================================================
# PRE-MARKET ANALYSIS
# =============================================================================

class PreMarketAnalyzer:
    """Pre-market (9:00-9:15) analysis."""

    def analyze_gaps(self, previous_close: float, pre_market_high: float,
                    pre_market_low: float, pre_market_ltp: float) -> Dict:
        """Analyze gap-up/gap-down."""
        gap_percent = (pre_market_ltp - previous_close) / previous_close * 100

        if gap_percent > 2:
            gap_type = 'STRONG_GAP_UP'
        elif gap_percent > 0.5:
            gap_type = 'GAP_UP'
        elif gap_percent < -2:
            gap_type = 'STRONG_GAP_DOWN'
        elif gap_percent < -0.5:
            gap_type = 'GAP_DOWN'
        else:
            gap_type = 'NO_GAP'

        return {
            'gap_type': gap_type,
            'gap_percent': gap_percent,
            'previous_close': previous_close,
            'pre_market_high': pre_market_high,
            'pre_market_low': pre_market_low
        }

    def opening_range_breakout(self, first_15min_high: float, first_15min_low: float,
                               current_price: float) -> Optional[Dict]:
        """Detect opening range breakout."""
        range_size = first_15min_high - first_15min_low

        if current_price > first_15min_high + (range_size * 0.3):
            return {
                'type': 'BULLISH_OBR',
                'target': first_15min_high + (range_size * 2),
                'stop': first_15min_low
            }

        if current_price < first_15min_low - (range_size * 0.3):
            return {
                'type': 'BEARISH_OBR',
                'target': first_15min_low - (range_size * 2),
                'stop': first_15min_high
            }

        return None

    def should_skip_open(self, gap_type: str) -> bool:
        """Determine if should skip first 15 minutes."""
        # Don't trade extreme gaps
        if 'STRONG' in gap_type:
            return True

        return False