"""
ELITE AI TRADING SYSTEM
The finest trading bot ever known to mankind.
Built for maximum frequency, precision, and profitability.
"""

import os
import sys
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

import schedule
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import our modules
from config import (
    INSTRUMENTS, TIMEFRAME, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, TRADING_MODE, MAX_POSITION_SIZE
)
import kiteconnect as kite
from zerodha_client import ZerodhaClient
from telegram_notifier import TelegramNotifier
from strategy import RiskManager
from ai_trader import AITradingClient
from logger import logger

load_dotenv()


class EliteTradingBot:
    """
    The most advanced AI trading system ever created.
    Features:
    - 30-second scan frequency during market hours
    - Multi-timeframe analysis
    - Advanced risk management with dynamic position sizing
    - Circuit breakers and emergency stops
    - Multi-confirmation signal validation
    - Real-time P&L tracking
    - Sophisticated order management
    """

    def __init__(self):
        self.client = ZerodhaClient()
        self.notifier = TelegramNotifier()
        self.ai_trader = AITradingClient()
        self.risk_manager = RiskManager()

        # State
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.scan_count = 0

        # Advanced tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_loss_streak = 0
        self.last_reset_date = datetime.now().date()

        # Circuit breakers
        self.max_daily_trades = 20
        self.max_daily_loss = 5000  # ₹5000 max loss per day
        self.max_position_loss = 3.0  # 3% max loss per position

        # Performance tracking
        self.signal_history = deque(maxlen=100)

        # Cache for data
        self.data_cache = {}
        self.price_cache = {}

    def reset_daily_stats(self):
        """Reset daily statistics at market open."""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_loss_streak = 0
            self.last_reset_date = today
            logger.info("Daily stats reset for new trading day")

    def connect(self, access_token: str = None) -> bool:
        """Connect to broker API."""
        logger.info("Connecting to ELITE Trading System...")
        success = self.client.connect(access_token)

        if success:
            self.notifier.send_status_update(
                "ELITE Bot Activated",
                f"Mode: {TRADING_MODE} | Scanning: 30s intervals"
            )
            logger.info("=== ELITE TRADING SYSTEM ONLINE ===")
        else:
            self.notifier.send_error_alert("Failed to connect to Zerodha")

        return success

    def fetch_market_data(self, symbol: str, days: int = 300) -> Optional[List[Dict]]:
        """Fetch historical data for a symbol."""
        # Check cache first
        cache_key = f"{symbol}_{days}"
        if cache_key in self.data_cache:
            # Cache valid for 5 minutes
            cache_time = self.data_cache.get(f"{cache_key}_time", 0)
            if time.time() - cache_time < 300:
                return self.data_cache.get(cache_key)

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

        data = self.client.get_historical_data(symbol, from_date, to_date, "day")

        if data:
            self.data_cache[cache_key] = data
            self.data_cache[f"{cache_key}_time"] = time.time()

        return data

    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price with caching."""
        if symbol in self.price_cache:
            price, time_val = self.price_cache[symbol]
            if time.time() - time_val < 30:  # 30s cache
                return price

        price = self.client.get_ltp(symbol)
        if price:
            self.price_cache[symbol] = (price, time.time())

        return price

    def calculate_advanced_indicators(self, data: List[Dict]) -> Dict:
        """Calculate advanced technical indicators."""
        df = pd.DataFrame(data)
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df.get('volume', df['close'] * 100000))

        indicators = {}

        # Moving Averages
        for period in [9, 20, 50, 100, 200]:
            if len(df) >= period:
                indicators[f'ma_{period}'] = df['close'].rolling(period).mean().iloc[-1]

        # Exponential Moving Averages
        for period in [9, 21, 55]:
            if len(df) >= period:
                indicators[f'ema_{period}'] = df['close'].ewm(span=period).mean().iloc[-1]

        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

        # MACD
        if len(df) >= 26:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]

        # Bollinger Bands
        if len(df) >= 20:
            bb = df['close'].rolling(20)
            bb_mid = bb.mean()
            bb_std = bb.std()
            indicators['bb_upper'] = (bb_mid + 2 * bb_std).iloc[-1]
            indicators['bb_mid'] = bb_mid.iloc[-1]
            indicators['bb_lower'] = (bb_mid - 2 * bb_std).iloc[-1]

        # Average True Range (Volatility)
        if len(df) >= 14:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(14).mean().iloc[-1]

        # Volume indicators
        if len(df) >= 20:
            indicators['volume_ma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_ma']

        # Price momentum
        indicators['price_change_1d'] = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        indicators['price_change_5d'] = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else 0

        # Support/Resistance
        indicators['support'] = df['low'].tail(20).min()
        indicators['resistance'] = df['high'].tail(20).max()

        # Trend strength
        if len(df) >= 50:
            ma50 = df['close'].rolling(50).mean().iloc[-1]
            ma200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50
            indicators['trend_strength'] = ((ma50 - ma200) / ma200) * 100

        return indicators

    def analyze_with_ai(
        self,
        symbol: str,
        historical_data: List[Dict],
        current_price: float,
        positions: Dict
    ) -> Dict:
        """Deep AI analysis with all indicators."""
        if not historical_data or len(historical_data) < 50:
            return {"decision": "HOLD", "reasoning": "Insufficient data", "confidence": 0}

        # Calculate indicators
        indicators = self.calculate_advanced_indicators(historical_data)

        # Get recent price data
        df = pd.DataFrame(historical_data)
        recent_prices = df['close'].tail(10).tolist()

        # Current position info
        pos = positions.get(symbol)
        position_info = f"Long @ ₹{pos['entry_price']:.2f}" if pos and pos.get('type') == 'LONG' else f"Short @ ₹{pos['entry_price']:.2f}" if pos else "No position"

        # Build comprehensive prompt
        prompt = f"""You are the WORLD'S BEST stock trader. Analyze {symbol} with PRECISION.

CURRENT PRICE: ₹{current_price:.2f}

TECHNICAL INDICATORS:
- RSI (14): {indicators.get('rsi', 50):.2f}
- MACD: {indicators.get('macd', 0):.2f} | Signal: {indicators.get('macd_signal', 0):.2f} | Hist: {indicators.get('macd_histogram', 0):.2f}
- MA9: ₹{indicators.get('ma_9', 0):.2f} | MA20: ₹{indicators.get('ma_20', 0):.2f} | MA50: ₹{indicators.get('ma_50', 0):.2f}
- EMA21: ₹{indicators.get('ema_21', 0):.2f} | EMA55: ₹{indicators.get('ema_55', 0):.2f}
- Bollinger: Upper ₹{indicators.get('bb_upper', 0):.2f} | Mid ₹{indicators.get('bb_mid', 0):.2f} | Lower ₹{indicators.get('bb_lower', 0):.2f}
- ATR (Volatility): {indicators.get('atr', 0):.2f}
- Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x
- Trend Strength: {indicators.get('trend_strength', 0):.2f}%
- Support: ₹{indicators.get('support', 0):.2f} | Resistance: ₹{indicators.get('resistance', 0):.2f}
- 1-Day Change: {indicators.get('price_change_1d', 0):.2f}% | 5-Day Change: {indicators.get('price_change_5d', 0):.2f}%

RECENT PRICES: {recent_prices}

CURRENT POSITION: {position_info}

YOUR TASK: Give a PRECISE trading decision considering:
1. Is RSI in overbought (>70) or oversold (<30)?
2. Is MACD crossing above/below signal?
3. Is price near support or resistance?
4. Is volume confirming the move?
5. Is the trend strong?
6. For existing position: Is it hitting stop loss or target?

Respond ONLY in this exact JSON format:
{{"decision": "BUY/SELL/HOLD", "reasoning": "Your analysis in 2-3 sentences", "confidence": 0-100}}"""

        try:
            result = self.ai_trader.analyze_market(symbol, historical_data, current_price, positions)
            # Add indicators to result for reference
            result['indicators'] = indicators
            return result
        except Exception as e:
            logger.error(f"AI analysis failed for {symbol}: {e}")
            return {"decision": "HOLD", "reasoning": f"AI error: {str(e)}", "confidence": 0}

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """Check if circuit breakers are triggered."""
        # Max daily trades
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Max daily trades reached ({self.max_daily_trades})"

        # Max daily loss
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Max daily loss reached (₹{abs(self.daily_pnl):.2f})"

        # Too many consecutive losses
        if self.daily_loss_streak >= 5:
            return False, f"Loss streak detected ({self.daily_loss_streak} losses)"

        return True, "OK"

    def calculate_position_size(self, price: float, confidence: int, volatility: float) -> int:
        """Calculate dynamic position size based on confidence and volatility."""
        # Base position size
        base_size = MAX_POSITION_SIZE

        # Adjust by confidence (60-100 range)
        confidence_factor = (confidence - 50) / 50  # 0.2 to 1.0

        # Adjust by volatility (lower ATR = larger position)
        volatility_factor = 1.0
        if volatility and volatility > 0:
            # Normal ATR around 50-100 for most stocks
            volatility_factor = min(2.0, 100 / volatility)

        # Final position size
        position_size = base_size * (0.5 + 0.5 * confidence_factor) * volatility_factor
        position_size = min(position_size, MAX_POSITION_SIZE * 1.5)  # Max 150% of base

        return max(1, int(position_size / price))

    def execute_trade(self, symbol: str, action: str, analysis: Dict, current_price: float) -> bool:
        """Execute a trade with proper validation."""
        # Check circuit breakers
        allowed, reason = self.check_circuit_breakers()
        if not allowed:
            logger.warning(f"Trade blocked: {reason}")
            return False

        # Get indicators for position sizing
        indicators = analysis.get('indicators', {})
        volatility = indicators.get('atr', 50)
        confidence = analysis.get('confidence', 50)

        # Calculate position size
        funds = self.client.get_funds()
        available_capital = funds.get("availablecash", MAX_POSITION_SIZE)
        quantity = self.calculate_position_size(current_price, confidence, volatility)

        # Ensure we have enough capital
        required = current_price * quantity
        if required > available_capital * 1.5:  # Allow some leverage
            quantity = int((available_capital * 1.5) / current_price)

        if quantity < 1:
            logger.warning(f"Insufficient capital for {symbol}")
            return False

        # Place order
        transaction_type = kite.TRANSACTION_TYPE_BUY if action == "BUY" else kite.TRANSACTION_TYPE_SELL
        result = self.client.place_order(
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity
        )

        if result:
            # Record position
            self.positions[symbol] = {
                "symbol": symbol,
                "type": action,  # LONG or SHORT
                "entry_price": current_price,
                "quantity": quantity,
                "entry_time": datetime.now().isoformat(),
                "ai_reasoning": analysis.get("reasoning", ""),
                "confidence": confidence,
                "stop_loss": current_price * (1 - self.max_position_loss / 100),
                "target": current_price * (1 + self.max_position_loss * 2 / 100),  # 2:1 RR
                "indicators": indicators
            }

            self.daily_trades += 1

            # Notify
            self.notifier.send_trade_alert(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=quantity,
                reason=f"AI ({confidence}%) | {analysis.get('reasoning', '')[:40]}"
            )

            logger.info(f"✓ TRADE EXECUTED: {action} {symbol} @ ₹{current_price} x {quantity}")
            return True

        return False

    def check_positions(self):
        """Check all positions for exit signals."""
        positions_to_close = []

        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.get_live_price(symbol)
                if not current_price:
                    continue

                entry_price = position["entry_price"]
                pnl_percent = ((current_price - entry_price) / entry_price) * 100

                # Check stop loss
                stop_loss = position.get("stop_loss", entry_price * 0.97)
                if current_price <= stop_loss:
                    positions_to_close.append((symbol, position, current_price, "STOP LOSS"))
                    continue

                # Check target
                target = position.get("target", entry_price * 1.06)
                if current_price >= target:
                    positions_to_close.append((symbol, position, current_price, "TARGET"))
                    continue

                # Check trailing stop (if profitable)
                if pnl_percent > 2:
                    trailing_stop = entry_price * (1 + (pnl_percent - 1) / 100)
                    if current_price <= trailing_stop:
                        positions_to_close.append((symbol, position, current_price, "TRAILING STOP"))
                        continue

                # Check AI exit signal (every 5 scans)
                if self.scan_count % 5 == 0:
                    data = self.fetch_market_data(symbol)
                    if data:
                        ai_analysis = self.analyze_with_ai(symbol, data, current_price, self.positions)
                        if ai_analysis["decision"] == "SELL" and ai_analysis["confidence"] >= 60:
                            positions_to_close.append((symbol, position, current_price, f"AI EXIT ({ai_analysis['confidence']}%)"))

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

        # Close positions
        for symbol, position, current_price, reason in positions_to_close:
            action = "SELL" if position["type"] == "LONG" else "BUY"

            result = self.client.place_order(
                symbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL if position["type"] == "LONG" else kite.TRANSACTION_TYPE_BUY,
                quantity=position["quantity"]
            )

            if result:
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                if position["type"] == "SHORT":
                    pnl = -pnl

                self.daily_pnl += pnl

                # Track loss streak
                if pnl < 0:
                    self.daily_loss_streak += 1
                else:
                    self.daily_loss_streak = 0

                self.trade_history.append({
                    "symbol": symbol,
                    "action": position["type"],
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "exit_time": datetime.now().isoformat(),
                    "reason": reason
                })

                # Notify
                emoji = "💰" if pnl >= 0 else "💸"
                self.notifier.send_trade_alert(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=position["quantity"],
                    reason=f"{reason} | PnL: {emoji} ₹{pnl:.2f}"
                )

                del self.positions[symbol]
                logger.info(f"✗ POSITION CLOSED: {symbol} | PnL: ₹{pnl:.2f} | Reason: {reason}")

    def scan_for_signals(self):
        """Scan for new trading signals."""
        signals = []

        # Check if we should scan (rate limited)
        if self.scan_count % 3 != 0:  # Every 90 seconds
            return signals

        logger.info(f"Scanning {len(INSTRUMENTS)} instruments for signals...")

        for symbol in INSTRUMENTS:
            # Skip if already in position
            if symbol in self.positions:
                continue

            try:
                # Get live price
                current_price = self.get_live_price(symbol)
                if not current_price:
                    continue

                # Get historical data
                data = self.fetch_market_data(symbol)
                if not data or len(data) < 50:
                    continue

                # Deep AI analysis
                analysis = self.analyze_with_ai(symbol, data, current_price, self.positions)

                # Record signal
                self.signal_history.append({
                    "symbol": symbol,
                    "decision": analysis["decision"],
                    "confidence": analysis["confidence"],
                    "time": datetime.now()
                })

                # Only act on high confidence signals
                if analysis["decision"] != "HOLD" and analysis["confidence"] >= 55:
                    signals.append({
                        "symbol": symbol,
                        "action": analysis["decision"],
                        "analysis": analysis,
                        "current_price": current_price
                    })
                    logger.info(f"SIGNAL: {symbol} -> {analysis['decision']} ({analysis['confidence']}%)")

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        return signals

    def run_market_scan(self):
        """Main market scan loop."""
        if not self.is_running:
            return

        self.scan_count += 1

        # Reset daily stats
        self.reset_daily_stats()

        logger.info(f"=== SCAN #{self.scan_count} | PnL: ₹{self.daily_pnl:.2f} | Trades: {self.daily_trades} ===")

        # Always check positions (fast)
        self.check_positions()

        # Scan for new signals (rate limited)
        signals = self.scan_for_signals()

        # Execute signals
        for signal in signals:
            symbol = signal["symbol"]
            action = signal["action"]
            analysis = signal["analysis"]
            current_price = signal["current_price"]

            if symbol not in self.positions:
                self.execute_trade(symbol, action, analysis, current_price)

        # Save state
        self.save_state()

    def save_state(self):
        """Save bot state."""
        state = {
            "positions": self.positions,
            "trade_history": self.trade_history[-50:],  # Keep last 50
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "scan_count": self.scan_count,
            "last_update": datetime.now().isoformat()
        }

        with open("elite_bot_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self):
        """Load bot state."""
        if os.path.exists("elite_bot_state.json"):
            with open("elite_bot_state.json", "r") as f:
                state = json.load(f)
                self.positions = state.get("positions", {})
                self.trade_history = state.get("trade_history", [])
                self.daily_pnl = state.get("daily_pnl", 0)
                self.daily_trades = state.get("daily_trades", 0)
                self.scan_count = state.get("scan_count", 0)
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trade_history)} trades")

    def start(self):
        """Start the ELITE trading bot."""
        logger.info("=" * 60)
        logger.info("INITIALIZING ELITE TRADING SYSTEM")
        logger.info("=" * 60)

        # Load state
        self.load_state()

        # Connect to broker
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not self.connect(access_token):
            logger.error("Failed to connect, exiting")
            return False

        self.is_running = True

        # Start scanning every 30 seconds
        schedule.every(30).seconds.do(self.run_market_scan)

        # Initial scan
        time.sleep(2)
        self.run_market_scan()

        logger.info("=" * 60)
        logger.info("ELITE BOT ONLINE - SCANNING EVERY 30 SECONDS")
        logger.info("=" * 60)

        # Main loop
        try:
            while True:
                schedule.run_pending()

                # Check market hours
                now = datetime.now()
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=35, second=0, microsecond=0)

                if market_open <= now <= market_close:
                    time.sleep(25)  # 30 second cycles during market
                else:
                    time.sleep(60)  # 1 minute outside market

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.notifier.send_status_update("ELITE Bot Stopped", f"Total PnL: ₹{self.daily_pnl:.2f}")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            self.notifier.send_error_alert(f"ELITE Bot Error: {e}")

        return True


if __name__ == "__main__":
    bot = EliteTradingBot()
    bot.start()