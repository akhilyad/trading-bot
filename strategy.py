"""
Moving Average Crossover Strategy
Simple, proven strategy for positional trading.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from config import SHORT_MA_PERIOD, LONG_MA_PERIOD, STOP_LOSS_PERCENT, TARGET_PROFIT_PERCENT
from logger import logger


class MACrossoverStrategy:
    """
    Moving Average Crossover Strategy.

    Buy when short MA crosses above long MA (golden cross).
    Sell when short MA crosses below long MA (death cross).
    """

    def __init__(self, short_period: int = SHORT_MA_PERIOD, long_period: int = LONG_MA_PERIOD):
        self.short_period = short_period
        self.long_period = long_period

    def calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate moving average."""
        return prices.rolling(window=period).mean()

    def analyze(self, historical_data: List[Dict]) -> Dict:
        """
        Analyze historical data and generate signals.

        Returns:
            Dict with keys: signal (BUY/SELL/HOLD), price, ma_short, ma_long, trend
        """
        if not historical_data or len(historical_data) < self.long_period + 5:
            logger.warning(f"Insufficient data for analysis (need {self.long_period + 5} candles)")
            return {"signal": "HOLD", "reason": "Insufficient data"}

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['close'] = pd.to_numeric(df['close'])

        # Calculate Moving Averages
        df['ma_short'] = self.calculate_ma(df['close'], self.short_period)
        df['ma_long'] = self.calculate_ma(df['close'], self.long_period)

        # Get latest values
        current_price = df['close'].iloc[-1]
        current_ma_short = df['ma_short'].iloc[-1]
        current_ma_long = df['ma_long'].iloc[-1]

        # Get previous values for crossover detection
        prev_ma_short = df['ma_short'].iloc[-2]
        prev_ma_long = df['ma_long'].iloc[-2]

        # Determine signal
        signal = "HOLD"
        reason = ""

        # Golden Cross: Short MA crosses above Long MA
        if prev_ma_short <= prev_ma_long and current_ma_short > current_ma_long:
            signal = "BUY"
            reason = f"Golden Cross: MA{self.short_period} (₹{current_ma_short:.2f}) crossed above MA{self.long_period} (₹{current_ma_long:.2f})"

        # Death Cross: Short MA crosses below Long MA
        elif prev_ma_short >= prev_ma_long and current_ma_short < current_ma_long:
            signal = "SELL"
            reason = f"Death Cross: MA{self.short_period} (₹{current_ma_short:.2f}) crossed below MA{self.long_period} (₹{current_ma_long:.2f})"

        # Trend check (only if we have enough data)
        else:
            if current_ma_short > current_ma_long:
                trend = "BULLISH"
                reason = f"Bullish trend: MA{self.short_period} > MA{self.long_period}"
            elif current_ma_short < current_ma_long:
                trend = "BEARISH"
                reason = f"Bearish trend: MA{self.short_period} < MA{self.long_period}"
            else:
                trend = "NEUTRAL"
                reason = "No clear trend"

        # Calculate stop loss and target
        stop_loss = current_price * (1 - STOP_LOSS_PERCENT / 100)
        target = current_price * (1 + TARGET_PROFIT_PERCENT / 100)

        result = {
            "signal": signal,
            "reason": reason,
            "current_price": current_price,
            "ma_short": current_ma_short,
            "ma_long": current_ma_long,
            "trend": trend if "trend" in locals() else "NEUTRAL",
            "stop_loss": stop_loss,
            "target": target,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Analysis: {signal} | Price: ₹{current_price:.2f} | MA{self.short_period}: ₹{current_ma_short:.2f} | MA{self.long_period}: ₹{current_ma_long:.2f}")

        return result


class RiskManager:
    """Handles position sizing and risk management."""

    def __init__(
        self,
        max_position_size: float = 1000,
        stop_loss_percent: float = STOP_LOSS_PERCENT,
        target_percent: float = TARGET_PROFIT_PERCENT
    ):
        self.max_position_size = max_position_size
        self.stop_loss_percent = stop_loss_percent
        self.target_percent = target_percent

    def calculate_quantity(self, price: float, available_capital: float) -> int:
        """Calculate quantity based on position size limit."""
        # Use the smaller of max position or available capital
        capital_to_use = min(self.max_position_size, available_capital)
        quantity = int(capital_to_use / price)
        return max(1, quantity)  # Minimum 1 share

    def should_take_trade(
        self,
        signal: str,
        current_price: float,
        entry_price: float = None,
        position_type: str = "LONG"
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be taken based on risk rules.

        Args:
            signal: BUY/SELL/HOLD
            current_price: Current market price
            entry_price: Price at which position was entered (for existing positions)
            position_type: LONG or SHORT

        Returns:
            (should_trade: bool, reason: str)
        """
        if signal == "HOLD":
            return False, "No signal"

        if entry_price:
            # For existing positions, check stop loss / target
            if position_type == "LONG":
                # Check stop loss
                sl_price = entry_price * (1 - self.stop_loss_percent / 100)
                if current_price <= sl_price:
                    return True, f"Stop loss hit: ₹{current_price:.2f} <= ₹{sl_price:.2f}"

                # Check target
                tp_price = entry_price * (1 + self.target_percent / 100)
                if current_price >= tp_price:
                    return True, f"Target hit: ₹{current_price:.2f} >= ₹{tp_price:.2f}"

        return True, "Signal received"

    def validate_position(self, position: Dict) -> bool:
        """Validate if a position needs to be closed."""
        if not position:
            return False

        current_price = position.get("current_price", 0)
        entry_price = position.get("entry_price", 0)
        position_type = position.get("type", "LONG")

        should_close, reason = self.should_take_trade(
            signal="SELL" if position_type == "LONG" else "BUY",
            current_price=current_price,
            entry_price=entry_price,
            position_type=position_type
        )

        if should_close and reason != "No signal":
            logger.info(f"Position close signal: {reason}")
            return True

        return False