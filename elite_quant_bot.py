"""
Onyx v2.0
The smartest, deadliest trading bot ever built.
Architecture: Multi-Timeframe | Regime Detection | Strategy Ensemble | Kelly Sizing | Portfolio Risk
"""

import os
import sys
import time
import json
import math
import random
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from copy import deepcopy

import schedule
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import our modules
from config import (
    INSTRUMENTS, TRADING_MODE, MAX_POSITION_SIZE
)
import kiteconnect as kite
from zerodha_client import ZerodhaClient
from telegram_notifier import TelegramNotifier
from ai_trader import AITradingClient
from logger import logger

load_dotenv()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketRegime:
    """Market regime classification."""
    name: str  # BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL, FII_BUYING, FII_SELLING
    confidence: float  # 0-100
    timestamp: datetime
    indicators: Dict = field(default_factory=dict)

@dataclass
class TradingSignal:
    """Complete trading signal."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strategy: str  # MOMENTUM, MEAN_REVERSION, STATISTICAL, OPTIONS
    confidence: int  # 0-100
    reasoning: str
    entry_price: float
    stop_loss: float
    target: float
    position_size: int
    timeframe: str
    risk_reward: float
    regime: str

@dataclass
class Position:
    """Position tracking."""
    symbol: str
    type: str  # LONG, SHORT
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float
    target: float
    strategy: str
    confidence: int
    regime: str
    indicators: Dict = field(default_factory=dict)
    scaled_entries: List[Dict] = field(default_factory=list)

@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    regime: str
    holding_period: int  # minutes
    confidence: int
    exit_reason: str

@dataclass
class PerformanceMetrics:
    """Performance tracking."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_time: float = 0.0


# =============================================================================
# DATA ENGINE - MULTI-TIMEFRAME
# =============================================================================

class DataEngine:
    """Fetches, processes, and caches market data across timeframes."""

    def __init__(self, client: ZerodhaClient):
        self.client = client
        self.cache = {}  # symbol -> {timeframe -> data}
        self.cache_expiry = 60  # seconds
        self.price_cache = {}  # symbol -> (price, timestamp)

    def fetch_ohlc(self, symbol: str, timeframe: str, days: int = 300) -> Optional[List[Dict]]:
        """Fetch OHLCV data for any timeframe."""
        cache_key = f"{symbol}_{timeframe}"

        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return data

        # Map timeframe to interval
        interval_map = {
            "1min": "1minute",
            "5min": "5minute",
            "15min": "15minute",
            "30min": "30minute",
            "1hour": "60minute",
            "1day": "day"
        }

        interval = interval_map.get(timeframe, "day")
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

        data = self.client.get_historical_data(symbol, from_date, to_date, interval)

        if data:
            self.cache[cache_key] = (data, time.time())

        return data

    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get cached live price."""
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if time.time() - timestamp < 30:  # 30s cache
                return price

        price = self.client.get_ltp(symbol)
        if price:
            self.price_cache[symbol] = (price, time.time())

        return price

    def get_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for all timeframes."""
        timeframes = ["1min", "5min", "15min", "1hour", "1day"]
        result = {}

        for tf in timeframes:
            data = self.fetch_ohlc(symbol, tf, days=300)
            if data and len(data) >= 20:
                df = pd.DataFrame(data)
                df['close'] = pd.to_numeric(df['close'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                df['volume'] = pd.to_numeric(df.get('volume', df['close'] * 100000))
                result[tf] = df

        return result

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators."""
        if df is None or len(df) < 20:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicators = {}

        # Moving Averages
        for period in [9, 20, 50, 100, 200]:
            if len(df) >= period:
                indicators[f'ma_{period}'] = close.rolling(period).mean().iloc[-1]
                indicators[f'ma_{period}_prev'] = close.rolling(period).mean().iloc[-2]

        # EMA
        for period in [9, 21, 55]:
            if len(df) >= period:
                indicators[f'ema_{period}'] = close.ewm(span=period).mean().iloc[-1]

        # RSI
        if len(df) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        if len(df) >= 26:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_prev'] = macd.iloc[-2]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_hist'] = (macd - signal).iloc[-1]
            indicators['macd_hist_prev'] = (macd - signal).iloc[-2]

        # Bollinger Bands
        if len(df) >= 20:
            bb = close.rolling(20)
            bb_mid = bb.mean()
            bb_std = bb.std()
            indicators['bb_upper'] = (bb_mid + 2 * bb_std).iloc[-1]
            indicators['bb_mid'] = bb_mid.iloc[-1]
            indicators['bb_lower'] = (bb_mid - 2 * bb_std).iloc[-1]
            # Position in BB
            indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'] + 0.001)

        # ATR (Volatility)
        if len(df) >= 14:
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(14).mean().iloc[-1]
            indicators['atr_percent'] = (indicators['atr'] / close.iloc[-1]) * 100

        # Volume
        if len(df) >= 20:
            indicators['volume_ma'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_ma']

        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            if len(df) >= period:
                indicators[f'change_{period}d'] = ((close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]) * 100

        # Support/Resistance
        indicators['support_20'] = low.tail(20).min()
        indicators['resistance_20'] = high.tail(20).max()
        indicators['support_50'] = low.tail(50).min()
        indicators['resistance_50'] = high.tail(50).max()

        # Trend direction
        if len(df) >= 50:
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50
            indicators['trend'] = 1 if ma50 > ma200 else -1
            indicators['trend_strength'] = ((ma50 - ma200) / ma200) * 100

        # Fractal (fractal support/resistance)
        if len(df) >= 5:
            high_fractal = high.rolling(3, center=True).max()
            low_fractal = low.rolling(3, center=True).min()
            indicators['fractal_high'] = high_fractal.iloc[-1]
            indicators['fractal_low'] = low_fractal.iloc[-1]

        # VWAP (if intraday data)
        if 'volume' in df.columns and len(df) >= 1:
            tp = (high + low + close) / 3
            cumvol = volume.cumsum()
            vwap = (tp * volume).cumsum() / cumvol
            indicators['vwap'] = vwap.iloc[-1]

        return indicators


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """Detects and tracks market regime for strategy selection."""

    def __init__(self):
        self.current_regime = None
        self.regime_history = deque(maxlen=100)
        self.fii_flow_cache = {}

    def detect_regime(self, nifty_data: pd.DataFrame, symbols_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """Detect current market regime using multiple indicators."""
        if nifty_data is None or len(nifty_data) < 50:
            return MarketRegime("SIDEWAYS", 50, datetime.now())

        close = nifty_data['close']
        volume = nifty_data.get('volume', pd.Series([1000000] * len(close)))

        # Trend detection (MA crossover)
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50

        trend_bull = ma50 > ma200
        trend_strength = abs((ma50 - ma200) / ma200) * 100

        # Volatility regime
        atr = self._calculate_atr(nifty_data)
        avg_atr = atr.rolling(20).mean().iloc[-1]
        current_atr_percent = (atr.iloc[-1] / close.iloc[-1]) * 100
        avg_atr_percent = (avg_atr / close.iloc[-1]) * 100

        high_vol = current_atr_percent > avg_atr_percent * 1.5
        low_vol = current_atr_percent < avg_atr_percent * 0.7

        # Range detection (sideways)
        range_20 = (close.tail(20).max() - close.tail(20).min()) / close.tail(20).mean() * 100
        is_sideways = range_20 < 5  # Less than 5% range in 20 days

        # Price position
        price_vs_ma50 = (close.iloc[-1] - ma50) / ma50 * 100

        # Volume trend
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

        # Determine regime
        regime_name = "SIDEWAYS"
        confidence = 50

        if trend_bull and trend_strength > 2:
            regime_name = "BULL"
            confidence = min(95, 50 + trend_strength * 10)

        elif not trend_bull and trend_strength > 2:
            regime_name = "BEAR"
            confidence = min(95, 50 + trend_strength * 10)

        elif is_sideways:
            regime_name = "SIDEWAYS"
            confidence = 70

        if high_vol:
            regime_name += "_HIGHVOL"
            confidence = min(95, confidence + 15)

        elif low_vol:
            regime_name += "_LOWVOL"
            confidence = min(95, confidence + 10)

        # Check FII flow (simulated from price action)
        fii_indicator = self._estimate_fii_flow(nifty_data)
        if fii_indicator > 0.3:
            regime_name = "FII_BUYING"
            confidence = min(95, confidence + 10)
        elif fii_indicator < -0.3:
            regime_name = "FII_SELLING"
            confidence = min(95, confidence + 10)

        regime = MarketRegime(
            name=regime_name,
            confidence=confidence,
            timestamp=datetime.now(),
            indicators={
                "trend_strength": trend_strength,
                "atr_percent": current_atr_percent,
                "range_20": range_20,
                "price_vs_ma50": price_vs_ma50,
                "volume_ratio": vol_ratio
            }
        )

        self.current_regime = regime
        self.regime_history.append(regime)

        return regime

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def _estimate_fii_flow(self, df: pd.DataFrame) -> float:
        """Estimate FII flow from price action (simplified)."""
        # In real implementation, use actual FII data
        # Here: net of price rise + volume expansion suggests buying
        close = df['close']
        volume = df.get('volume', pd.Series([1000000] * len(close)))

        price_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        vol_change = volume.iloc[-1] / volume.rolling(5).mean().iloc[-1]

        return (price_change * 100) + (vol_change - 1) * 10

    def get_strategy_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get strategy weights based on regime."""
        weights = {
            "MOMENTUM": 0.25,
            "MEAN_REVERSION": 0.25,
            "STATISTICAL": 0.25,
            "OPTIONS": 0.25
        }

        regime_name = regime.name

        if "BULL" in regime_name and "HIGHVOL" not in regime_name:
            weights = {"MOMENTUM": 0.50, "MEAN_REVERSION": 0.20, "STATISTICAL": 0.20, "OPTIONS": 0.10}

        elif "BULL" in regime_name and "HIGHVOL" in regime_name:
            weights = {"MOMENTUM": 0.40, "MEAN_REVERSION": 0.30, "STATISTICAL": 0.20, "OPTIONS": 0.10}

        elif "BEAR" in regime_name:
            weights = {"MOMENTUM": 0.15, "MEAN_REVERSION": 0.40, "STATISTICAL": 0.25, "OPTIONS": 0.20}

        elif "SIDEWAYS" in regime_name:
            weights = {"MOMENTUM": 0.10, "MEAN_REVERSION": 0.45, "STATISTICAL": 0.35, "OPTIONS": 0.10}

        elif "HIGHVOL" in regime_name:
            weights = {"MOMENTUM": 0.35, "MEAN_REVERSION": 0.35, "STATISTICAL": 0.20, "OPTIONS": 0.10}

        elif "LOWVOL" in regime_name:
            weights = {"MOMENTUM": 0.20, "MEAN_REVERSION": 0.30, "STATISTICAL": 0.40, "OPTIONS": 0.10}

        elif "FII_BUYING" in regime_name:
            weights = {"MOMENTUM": 0.55, "MEAN_REVERSION": 0.15, "STATISTICAL": 0.20, "OPTIONS": 0.10}

        elif "FII_SELLING" in regime_name:
            weights = {"MOMENTUM": 0.10, "MEAN_REVERSION": 0.40, "STATISTICAL": 0.30, "OPTIONS": 0.20}

        return weights


# =============================================================================
# STRATEGY ENSEMBLE
# =============================================================================

class StrategyEnsemble:
    """Multi-strategy system with AI-powered signal generation."""

    def __init__(self, ai_client: AITradingClient):
        self.ai_client = ai_client

    def generate_signals(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pd.DataFrame],
        regime: MarketRegime,
        existing_position: Optional[Position] = None
    ) -> List[TradingSignal]:
        """Generate signals from all strategies."""
        signals = []

        # Get indicators for each timeframe
        indicators = {}
        for tf, df in multi_tf_data.items():
            if df is not None:
                indicators[tf] = DataEngine(None).calculate_indicators(df)

        # Strategy 1: Momentum
        momentum_signal = self._momentum_strategy(symbol, multi_tf_data, indicators, regime, existing_position)
        if momentum_signal:
            signals.append(momentum_signal)

        # Strategy 2: Mean Reversion
        mr_signal = self._mean_reversion_strategy(symbol, multi_tf_data, indicators, regime, existing_position)
        if mr_signal:
            signals.append(mr_signal)

        # Strategy 3: Statistical
        stat_signal = self._statistical_strategy(symbol, multi_tf_data, indicators, regime, existing_position)
        if stat_signal:
            signals.append(stat_signal)

        return signals

    def _momentum_strategy(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pd.DataFrame],
        indicators: Dict,
        regime: MarketRegime,
        existing_position: Optional[Position]
    ) -> Optional[TradingSignal]:
        """Momentum/trend-following strategy."""
        # Primary: 15min and 1hour for entries
        df_15m = multi_tf_data.get("15min")
        df_1h = multi_tf_data.get("1day")

        if df_15m is None or df_1h is None:
            return None

        close = df_15m['close']
        current_price = close.iloc[-1]

        # Indicators
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20

        # Trend check (1h)
        df_daily = multi_tf_data.get("1day")
        if df_daily is not None:
            daily_ma50 = df_daily['close'].rolling(50).mean().iloc[-1]
            daily_ma200 = df_daily['close'].rolling(200).mean().iloc[-1] if len(df_daily) >= 200 else daily_ma50
            trend_bull = daily_ma50 > daily_ma200
        else:
            trend_bull = True

        # Entry conditions (Momentum)
        buy_signal = (
            current_price > ma20 and
            current_price > ma50 and
            trend_bull and
            "BEAR" not in regime.name
        )

        sell_signal = (
            current_price < ma20 and
            current_price < ma50 and
            not trend_bull and
            "BULL" not in regime.name
        )

        if buy_signal:
            # Calculate stops and target
            atr = df_15m['high'] - df_15m['low']
            atr_value = atr.rolling(14).mean().iloc[-1]

            stop_loss = current_price - (2 * atr_value)
            target = current_price + (3 * atr_value)  # 1.5:1 R:R minimum

            return TradingSignal(
                symbol=symbol,
                action="BUY",
                strategy="MOMENTUM",
                confidence=75,
                reasoning=f"Momentum breakout: Price above MA20/MA50, trend aligned",
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="15min",
                risk_reward=1.5,
                regime=regime.name
            )

        elif sell_signal:
            atr = df_15m['high'] - df_15m['low']
            atr_value = atr.rolling(14).mean().iloc[-1]

            stop_loss = current_price + (2 * atr_value)
            target = current_price - (3 * atr_value)

            return TradingSignal(
                symbol=symbol,
                action="SELL",
                strategy="MOMENTUM",
                confidence=75,
                reasoning=f"Momentum breakdown: Price below MA20/MA50, downtrend",
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="15min",
                risk_reward=1.5,
                regime=regime.name
            )

        return None

    def _mean_reversion_strategy(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pd.DataFrame],
        indicators: Dict,
        regime: MarketRegime,
        existing_position: Optional[Position]
    ) -> Optional[TradingSignal]:
        """Mean reversion/oscillator strategy."""
        df_5m = multi_tf_data.get("5min")
        df_15m = multi_tf_data.get("15min")

        if df_5m is None:
            return None

        close = df_5m['close']
        current_price = close.iloc[-1]

        # RSI for mean reversion
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # Bollinger Bands
        bb = close.rolling(20)
        bb_mid = bb.mean()
        bb_std = bb.std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_pos = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 0.001)

        # Entry conditions (Mean Reversion)
        # Buy when oversold, sell when overbought
        buy_signal = rsi < 35 and bb_pos < 0.2
        sell_signal = rsi > 65 and bb_pos > 0.8

        # Don't trade momentum regimes with mean reversion
        if "BULL" in regime.name and regime.confidence > 70:
            return None

        if buy_signal:
            atr = df_5m['high'] - df_5m['low']
            atr_value = atr.rolling(14).mean().iloc[-1]

            stop_loss = current_price - (1.5 * atr_value)
            target = current_price + (2 * atr_value)  # 1.33:1 R:R

            return TradingSignal(
                symbol=symbol,
                action="BUY",
                strategy="MEAN_REVERSION",
                confidence=70,
                reasoning=f"Oversold: RSI={rsi:.0f}, BB position={bb_pos:.2f}",
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="5min",
                risk_reward=1.33,
                regime=regime.name
            )

        elif sell_signal:
            atr = df_5m['high'] - df_5m['low']
            atr_value = atr.rolling(14).mean().iloc[-1]

            stop_loss = current_price + (1.5 * atr_value)
            target = current_price - (2 * atr_value)

            return TradingSignal(
                symbol=symbol,
                action="SELL",
                strategy="MEAN_REVERSION",
                confidence=70,
                reasoning=f"Overbought: RSI={rsi:.0f}, BB position={bb_pos:.2f}",
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="5min",
                risk_reward=1.33,
                regime=regime.name
            )

        return None

    def _statistical_strategy(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pd.DataFrame],
        indicators: Dict,
        regime: MarketRegime,
        existing_position: Optional[Position]
    ) -> Optional[TradingSignal]:
        """Statistical/arbitrage strategy using correlations."""
        df_daily = multi_tf_data.get("1day")

        if df_daily is None or len(df_daily) < 30:
            return None

        # For Indian stocks, we use sector correlations
        # Simple version: check for statistical outliers in price

        close = df_daily['close']
        returns = close.pct_change()

        # Z-score of current price vs 20-day mean
        ma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        z_score = (close.iloc[-1] - ma20) / std20

        # Buy when significantly below mean (reversion to mean expected)
        if z_score < -2.0:
            stop_loss = close.iloc[-1] * 0.95
            target = ma20  # Expect reversion to mean

            return TradingSignal(
                symbol=symbol,
                action="BUY",
                strategy="STATISTICAL",
                confidence=65,
                reasoning=f"Statistical outlier: Z-score={z_score:.2f}, expect reversion",
                entry_price=close.iloc[-1],
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="1day",
                risk_reward=1.5,
                regime=regime.name
            )

        elif z_score > 2.0:
            stop_loss = close.iloc[-1] * 1.05
            target = ma20

            return TradingSignal(
                symbol=symbol,
                action="SELL",
                strategy="STATISTICAL",
                confidence=65,
                reasoning=f"Statistical outlier: Z-score={z_score:.2f}, expect reversion",
                entry_price=close.iloc[-1],
                stop_loss=stop_loss,
                target=target,
                position_size=0,
                timeframe="1day",
                risk_reward=1.5,
                regime=regime.name
            )

        return None


# =============================================================================
# RISK MANAGEMENT ENGINE
# =============================================================================

class RiskManager:
    """Advanced risk management with Kelly Criterion and portfolio-level controls."""

    def __init__(self):
        self.max_portfolio_risk = 0.02  # 2% of portfolio per day
        self.max_position_risk = 0.01   # 1% per position
        self.max_total_exposure = 0.60   # Max 60% invested
        self.max_correlation_exposure = 0.30  # Max 30% in correlated positions
        self.kelly_fraction = 0.25  # Use 25% of Kelly (conservative)

        # Performance tracking
        self.win_rate = 0.50
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.trade_history = deque(maxlen=100)

    def calculate_kelly_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: int,
        available_capital: float
    ) -> int:
        """Calculate position size using Kelly Criterion."""
        # Update win rate from history
        self._update_performance()

        # Kelly formula: K% = W - (W - R) / R where R = avg_win / avg_loss
        if self.avg_loss > 0:
            reward_ratio = self.avg_win / abs(self.avg_loss)
        else:
            reward_ratio = 1.5

        win_rate = self.win_rate

        # Kelly percentage
        if reward_ratio > 0:
            kelly_pct = win_rate - ((win_rate * reward_ratio - win_rate) / reward_ratio)
        else:
            kelly_pct = 0

        # Adjust by confidence
        confidence_factor = 0.5 + (confidence / 200)  # 0.5 to 1.0
        adjusted_kelly = kelly_pct * self.kelly_fraction * confidence_factor

        # Cap at max position risk
        max_kelly = self.max_position_risk
        adjusted_kelly = min(adjusted_kelly, max_kelly)

        # Calculate position size
        position_value = available_capital * adjusted_kelly
        quantity = int(position_value / entry_price)

        return max(1, quantity)

    def _update_performance(self):
        """Update performance metrics from trade history."""
        if not self.trade_history:
            return

        wins = [t.pnl for t in self.trade_history if t.pnl > 0]
        losses = [t.pnl for t in self.trade_history if t.pnl < 0]

        if wins:
            self.avg_win = sum(wins) / len(wins)
        if losses:
            self.avg_loss = sum(losses) / len(losses)

        if wins or losses:
            self.win_rate = len(wins) / (len(wins) + len(losses))

    def check_portfolio_risk(
        self,
        current_exposure: float,
        positions: Dict[str, Position],
        trade_pnl: float
    ) -> Tuple[bool, str]:
        """Check portfolio-level risk limits."""
        # Daily loss limit
        if trade_pnl < -MAX_POSITION_SIZE * 0.1:  # 10% of max position in losses
            return False, "Daily loss limit reached"

        # Max exposure
        if current_exposure >= self.max_total_exposure:
            return False, "Max portfolio exposure reached"

        # Max positions
        if len(positions) >= 10:
            return False, "Max number of positions reached"

        return True, "OK"

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        volatility_regime: str,
        strategy: str
    ) -> float:
        """Calculate dynamic stop loss based on volatility."""
        # Base ATR multiplier
        atr_mult = {
            "HIGH_VOL": 3.0,
            "LOW_VOL": 1.5,
            "NORMAL": 2.0
        }.get(volatility_regime, 2.0)

        # Strategy-specific adjustments
        if strategy == "MEAN_REVERSION":
            atr_mult *= 1.2
        elif strategy == "MOMENTUM":
            atr_mult *= 0.8

        return entry_price - (atr_mult * atr)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_min: float = 2.0
    ) -> float:
        """Calculate profit target based on risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        return entry_price + (risk * risk_reward_min)


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """Smart order execution with scaling and time rules."""

    def __init__(self, client: ZerodhaClient, notifier: TelegramNotifier):
        self.client = client
        self.notifier = notifier
        self.pending_orders = {}

    def execute_signal(
        self,
        signal: TradingSignal,
        available_capital: float,
        mode: str = "paper"
    ) -> Optional[Position]:
        """Execute a trading signal with smart execution."""
        if mode == "paper":
            logger.info(f"[PAPER] Would execute: {signal.action} {signal.symbol} @ ₹{signal.entry_price}")
            # Create paper position
            position = Position(
                symbol=signal.symbol,
                type=signal.action,
                entry_price=signal.entry_price,
                quantity=signal.position_size,
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                target=signal.target,
                strategy=signal.strategy,
                confidence=signal.confidence,
                regime=signal.regime,
                indicators={}
            )

            self.notifier.send_trade_alert(
                symbol=signal.symbol,
                action=signal.action,
                price=signal.entry_price,
                quantity=signal.position_size,
                reason=f"PAPER: {signal.strategy} ({signal.confidence}%) {signal.reasoning[:30]}"
            )

            return position

        # Live execution
        try:
            transaction_type = kite.TRANSACTION_TYPE_BUY if signal.action == "BUY" else kite.TRANSACTION_TYPE_SELL

            result = self.client.place_order(
                symbol=signal.symbol,
                transaction_type=transaction_type,
                quantity=signal.position_size
            )

            if result:
                position = Position(
                    symbol=signal.symbol,
                    type=signal.action,
                    entry_price=signal.entry_price,
                    quantity=signal.position_size,
                    entry_time=datetime.now(),
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    regime=signal.regime,
                    indicators={}
                )

                self.notifier.send_trade_alert(
                    symbol=signal.symbol,
                    action=signal.action,
                    price=signal.entry_price,
                    quantity=signal.position_size,
                    reason=f"LIVE: {signal.strategy} ({signal.confidence}%)"
                )

                return position

        except Exception as e:
            logger.error(f"Execution failed: {e}")

        return None

    def close_position(
        self,
        position: Position,
        current_price: float,
        reason: str,
        mode: str = "paper"
    ) -> Optional[Trade]:
        """Close a position and record the trade."""
        if mode == "paper":
            pnl = (current_price - position.entry_price) * position.quantity
            if position.type == "SHORT":
                pnl = -pnl

            trade = Trade(
                symbol=position.symbol,
                action=position.type,
                entry_price=position.entry_price,
                exit_price=current_price,
                quantity=position.quantity,
                pnl=pnl,
                pnl_percent=(pnl / (position.entry_price * position.quantity)) * 100,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                strategy=position.strategy,
                regime=position.regime,
                holding_period=int((datetime.now() - position.entry_time).total_seconds() / 60),
                confidence=position.confidence,
                exit_reason=reason
            )

            emoji = "💰" if pnl >= 0 else "💸"
            self.notifier.send_trade_alert(
                symbol=position.symbol,
                action="SELL" if position.type == "LONG" else "BUY",
                price=current_price,
                quantity=position.quantity,
                reason=f"{reason} | PnL: {emoji}₹{pnl:.2f}"
            )

            return trade

        # Live close
        try:
            transaction_type = kite.TRANSACTION_TYPE_SELL if position.type == "LONG" else kite.TRANSACTION_TYPE_BUY

            result = self.client.place_order(
                symbol=position.symbol,
                transaction_type=transaction_type,
                quantity=position.quantity
            )

            if result:
                pnl = (current_price - position.entry_price) * position.quantity
                if position.type == "SHORT":
                    pnl = -pnl

                trade = Trade(
                    symbol=position.symbol,
                    action=position.type,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    pnl_percent=(pnl / (position.entry_price * position.quantity)) * 100,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    strategy=position.strategy,
                    regime=position.regime,
                    holding_period=int((datetime.now() - position.entry_time).total_seconds() / 60),
                    confidence=position.confidence,
                    exit_reason=reason
                )

                return trade

        except Exception as e:
            logger.error(f"Close failed: {e}")

        return None


# =============================================================================
# BACKTEST MODULE
# =============================================================================

class BacktestEngine:
    """Backtesting with walk-forward optimization."""

    def __init__(self):
        self.results = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func,
        initial_capital: float = 100000,
        commission: float = 0.001
    ) -> Dict:
        """Run simple backtest on historical data."""
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]

        for i in range(50, len(data)):
            current_price = data['close'].iloc[i]

            # Generate signal
            signal = strategy_func(data.iloc[:i+1])

            if signal and position is None:
                # Entry
                quantity = int(capital * 0.1 / current_price)
                position = {
                    'entry': current_price,
                    'quantity': quantity,
                    'entry_idx': i
                }
                capital -= (current_price * quantity * (1 + commission))

            elif position:
                # Check exit
                exit_signal = strategy_func(data.iloc[:i+1], exit_only=True)

                if exit_signal or i - position['entry_idx'] > 20:  # Max 20 days
                    pnl = (current_price - position['entry']) * position['quantity']
                    pnl -= (current_price * position['quantity'] * commission)
                    capital += (current_price * position['quantity'])

                    trades.append({
                        'pnl': pnl,
                        'return': pnl / (position['entry'] * position['quantity'])
                    })
                    position = None

            equity_curve.append(capital)

        return {
            'trades': trades,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
            'avg_return': sum([t['return'] for t in trades]) / len(trades) if trades else 0
        }

    def monte_carlo_simulation(
        self,
        trades: List[Trade],
        simulations: int = 1000
    ) -> Dict:
        """Run Monte Carlo simulation on trade outcomes."""
        if not trades:
            return {}

        returns = [t.pnl_percent for t in trades]

        results = []
        for _ in range(simulations):
            # Resample with replacement
            sample = random.choices(returns, k=len(returns))
            results.append(sum(sample))

        results.sort()

        return {
            'worst_5%': results[int(simulations * 0.05)],
            'best_5%': results[int(simulations * 0.95)],
            'median': results[simulations // 2],
            'mean': sum(results) / simulations
        }


# =============================================================================
# LEARNING MODULE
# =============================================================================

class LearningModule:
    """Self-improvement based on trade outcomes."""

    def __init__(self):
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        self.regime_performance = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.confidence_calibration = []

    def record_trade(self, trade: Trade):
        """Record trade for learning."""
        # Strategy performance
        self.strategy_performance[trade.strategy]['total_pnl'] += trade.pnl
        if trade.pnl > 0:
            self.strategy_performance[trade.strategy]['wins'] += 1
        else:
            self.strategy_performance[trade.strategy]['losses'] += 1

        # Regime performance
        if trade.pnl > 0:
            self.regime_performance[trade.regime]['wins'] += 1
        else:
            self.regime_performance[trade.regime]['losses'] += 1

        # Confidence calibration
        if trade.pnl > 0 and trade.confidence > 70:
            self.confidence_calibration.append(1)
        elif trade.pnl < 0 and trade.confidence > 70:
            self.confidence_calibration.append(0)

    def get_best_strategy_for_regime(self, regime: str) -> str:
        """Get best performing strategy for current regime."""
        perf = self.regime_performance.get(regime, {})
        wins = perf.get('wins', 0)
        losses = perf.get('losses', 0)

        if wins + losses == 0:
            return "MOMENTUM"

        win_rate = wins / (wins + losses)

        # Return best strategy
        if win_rate > 0.6:
            return "MOMENTUM"
        elif win_rate > 0.5:
            return "MEAN_REVERSION"
        else:
            return "STATISTICAL"

    def get_adjusted_confidence(self, base_confidence: int, strategy: str, regime: str) -> int:
        """Adjust confidence based on historical performance."""
        # Get strategy performance
        strat_perf = self.strategy_performance.get(strategy, {})
        strat_wins = strat_perf.get('wins', 0)
        strat_losses = strat_perf.get('losses', 0)

        if strat_wins + strat_losses > 5:
            strat_winrate = strat_wins / (strat_wins + strat_losses)
            # Adjust by up to 15 points
            adjustment = (strat_winrate - 0.5) * 30
            return max(30, min(95, base_confidence + int(adjustment)))

        return base_confidence


# =============================================================================
# MAIN ELITE BOT
# =============================================================================

class EliteQuantBot:
    """
    The most advanced trading bot ever built.
    Combines regime detection, multi-strategy ensemble, Kelly sizing, and continuous learning.
    """

    def __init__(self):
        # Core components
        self.client = ZerodhaClient()
        self.notifier = TelegramNotifier()
        self.ai_client = AITradingClient()

        # Trading engines
        self.data_engine = DataEngine(self.client)
        self.regime_detector = RegimeDetector()
        self.strategy_ensemble = StrategyEnsemble(self.ai_client)
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine(self.client, self.notifier)
        self.backtest_engine = BacktestEngine()
        self.learning_module = LearningModule()

        # State
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.is_running = False
        self.scan_count = 0

        # Circuit breakers
        self.max_daily_trades = 25
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = 8000  # ₹8000
        self.cooloff_until = None

        # Config
        self.scan_interval = 30  # seconds
        self.market_open = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        self.market_close = datetime.now().replace(hour=15, minute=35, second=0, microsecond=0)

    def connect(self) -> bool:
        """Initialize connections."""
        logger.info("=" * 60)
        logger.info("Onyx v2.0 - INITIALIZING")
        logger.info("=" * 60)

        # Connect to broker
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not self.client.connect(access_token):
            logger.error("Failed to connect to broker")
            return False

        self.notifier.send_status_update(
            "ELITE v2.0 Activated",
            f"Mode: {TRADING_MODE} | Scan: {self.scan_interval}s"
        )

        logger.info("All systems online")
        return True

    def run_scan(self):
        """Main scan loop - runs every scan_interval."""
        if not self.is_running:
            return

        self.scan_count += 1

        # Check circuit breakers
        if self.cooloff_until and datetime.now() < self.cooloff_until:
            logger.info(f"Cooloff mode, waiting until {self.cooloff_until}")
            return

        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning("Daily loss limit reached - cooling off")
            self.cooloff_until = datetime.now() + timedelta(hours=4)
            self.notifier.send_error_alert(f"Daily loss limit reached: ₹{self.daily_pnl}")
            return

        logger.info(f"\n{'='*50}")
        logger.info(f"SCAN #{self.scan_count} | PnL: ₹{self.daily_pnl:.2f} | Trades: {self.daily_trades}")
        logger.info(f"{'='*50}")

        # Step 1: Check positions for exits
        self._check_positions()

        # Step 2: Detect market regime
        nifty_data = self._get_nifty_data()
        if nifty_data is not None:
            regime = self.regime_detector.detect_regime(nifty_data, {})
            logger.info(f"Regime: {regime.name} ({regime.confidence}%)")
        else:
            regime = MarketRegime("SIDEWAYS", 50, datetime.now())

        # Step 3: Scan for new signals (rate limited to save API)
        if self.scan_count % 3 == 0:
            self._scan_for_signals(regime)

        # Step 4: Save state
        self._save_state()

    def _get_nifty_data(self) -> Optional[pd.DataFrame]:
        """Get NIFTY index data for regime detection."""
        try:
            data = self.data_engine.fetch_ohlc("NIFTY 50", "1day", days=300)
            if data:
                df = pd.DataFrame(data)
                df['close'] = pd.to_numeric(df['close'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                return df
        except Exception as e:
            logger.error(f"Failed to get NIFTY data: {e}")
        return None

    def _check_positions(self):
        """Check all positions for exit conditions."""
        positions_to_close = []

        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.data_engine.get_live_price(symbol)
                if not current_price:
                    continue

                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100

                # Stop loss hit
                if position.type == "LONG" and current_price <= position.stop_loss:
                    positions_to_close.append((symbol, position, current_price, "STOP_LOSS"))
                    continue
                elif position.type == "SHORT" and current_price >= position.stop_loss:
                    positions_to_close.append((symbol, position, current_price, "STOP_LOSS"))
                    continue

                # Target hit
                if position.type == "LONG" and current_price >= position.target:
                    positions_to_close.append((symbol, position, current_price, "TARGET"))
                    continue
                elif position.type == "SHORT" and current_price <= position.target:
                    positions_to_close.append((symbol, position, current_price, "TARGET"))
                    continue

                # Trailing stop (if 3%+ profit)
                if pnl_percent > 3:
                    trailing_stop = position.entry_price * (1 + (pnl_percent - 1.5) / 100)
                    if position.type == "LONG" and current_price <= trailing_stop:
                        positions_to_close.append((symbol, position, current_price, "TRAILING_STOP"))
                        continue

                # Time-based exit (max 45 min for intraday)
                holding_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
                if holding_minutes > 45 and "SIDEWAYS" in position.regime:
                    positions_to_close.append((symbol, position, current_price, "TIME_EXIT"))

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

        # Execute closes
        for symbol, position, current_price, reason in positions_to_close:
            trade = self.execution_engine.close_position(position, current_price, reason, TRADING_MODE)
            if trade:
                self._process_closed_trade(symbol, trade)

    def _scan_for_signals(self, regime: MarketRegime):
        """Scan all instruments for trading signals."""
        if self.daily_trades >= self.max_daily_trades:
            logger.info("Max daily trades reached")
            return

        # Get available capital
        funds = self.client.get_funds()
        available_capital = funds.get("availablecash", MAX_POSITION_SIZE)

        # Get strategy weights
        strategy_weights = self.regime_detector.get_strategy_weights(regime)

        logger.info(f"Scanning {len(INSTRUMENTS)} instruments...")

        for symbol in INSTRUMENTS:
            # Skip if already in position
            if symbol in self.positions:
                continue

            try:
                # Get multi-timeframe data
                multi_tf = self.data_engine.get_multiple_timeframes(symbol)
                if not multi_tf:
                    continue

                current_price = self.data_engine.get_live_price(symbol)
                if not current_price:
                    continue

                # Generate signals from ensemble
                existing_pos = self.positions.get(symbol)
                signals = self.strategy_ensemble.generate_signals(
                    symbol, multi_tf, regime, existing_pos
                )

                # Select best signal
                best_signal = None
                for signal in signals:
                    # Adjust confidence based on learning
                    signal.confidence = self.learning_module.get_adjusted_confidence(
                        signal.confidence, signal.strategy, regime.name
                    )

                    if signal.confidence >= 55 and signal.risk_reward >= 1.3:
                        if best_signal is None or signal.confidence > best_signal.confidence:
                            best_signal = signal

                if best_signal:
                    # Calculate position size
                    quantity = self.risk_manager.calculate_kelly_position_size(
                        best_signal.entry_price,
                        best_signal.stop_loss,
                        best_signal.confidence,
                        available_capital
                    )
                    best_signal.position_size = quantity

                    # Check risk
                    can_trade, reason = self.risk_manager.check_portfolio_risk(
                        len(self.positions) * 0.1,
                        self.positions,
                        self.daily_pnl
                    )

                    if can_trade:
                        # Execute
                        position = self.execution_engine.execute_signal(
                            best_signal, available_capital, TRADING_MODE
                        )
                        if position:
                            self.positions[symbol] = position
                            self.daily_trades += 1
                            logger.info(f"✓ ENTERED: {symbol} {best_signal.action} @ ₹{current_price}")

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

    def _process_closed_trade(self, symbol: str, trade: Trade):
        """Process closed trade."""
        self.trade_history.append(trade)
        self.daily_pnl += trade.pnl
        self.risk_manager.trade_history.append(trade)
        self.learning_module.record_trade(trade)

        if symbol in self.positions:
            del self.positions[symbol]

        logger.info(f"✗ EXIT: {symbol} | PnL: ₹{trade.pnl:.2f} | {trade.exit_reason}")

    def _save_state(self):
        """Save bot state."""
        state = {
            'positions': {s: {
                'symbol': p.symbol,
                'type': p.type,
                'entry_price': p.entry_price,
                'quantity': p.quantity,
                'entry_time': p.entry_time.isoformat(),
                'stop_loss': p.stop_loss,
                'target': p.target,
                'strategy': p.strategy,
                'confidence': p.confidence,
                'regime': p.regime
            } for s, p in self.positions.items()},
            'trade_history': [{
                'symbol': t.symbol,
                'pnl': t.pnl,
                'strategy': t.strategy,
                'regime': t.regime,
                'exit_reason': t.exit_reason
            } for t in self.trade_history[-50:]],
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'scan_count': self.scan_count
        }

        with open("elite_v2_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self):
        """Load previous state."""
        if os.path.exists("elite_v2_state.json"):
            with open("elite_v2_state.json", "r") as f:
                state = json.load(f)
                self.daily_pnl = state.get('daily_pnl', 0)
                self.daily_trades = state.get('daily_trades', 0)
                self.scan_count = state.get('scan_count', 0)
                logger.info("State loaded")

    def start(self):
        """Start the bot."""
        logger.info("Starting Onyx v2.0...")

        self.load_state()

        if not self.connect():
            return

        self.is_running = True

        # Start scanning
        schedule.every(self.scan_interval).seconds.do(self.run_scan)

        # Initial scan
        time.sleep(2)
        self.run_scan()

        logger.info(f"ELITE v2.0 ONLINE - Scanning every {self.scan_interval}s")

        # Main loop
        try:
            while True:
                schedule.run_pending()

                now = datetime.now()
                if self.market_open <= now <= self.market_close:
                    time.sleep(self.scan_interval - 1)
                else:
                    time.sleep(60)  # Check every minute outside market

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.notifier.send_status_update("ELITE v2.0 Stopped", f"PnL: ₹{self.daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Error: {e}")
            self.notifier.send_error_alert(f"Error: {e}")

    def get_performance_report(self) -> PerformanceMetrics:
        """Generate performance report."""
        if not self.trade_history:
            return PerformanceMetrics()

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl < 0]

        return PerformanceMetrics(
            total_trades=len(self.trade_history),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(t.pnl for t in self.trade_history),
            avg_win=sum(t.pnl for t in wins) / len(wins) if wins else 0,
            avg_loss=sum(t.pnl for t in losses) / len(losses) if losses else 0,
            win_rate=len(wins) / len(self.trade_history),
            profit_factor=abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses else 0,
            avg_holding_time=sum(t.holding_period for t in self.trade_history) / len(self.trade_history)
        )


if __name__ == "__main__":
    bot = EliteQuantBot()
    bot.start()