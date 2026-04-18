"""
AI-Powered Trading Bot
Uses Claude/Minimax API to analyze markets and make trading decisions.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import schedule
import pandas as pd
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

# Load environment variables
load_dotenv()


class AITradingBot:
    """AI-powered trading bot using Claude/Minimax API."""

    def __init__(self):
        self.client = ZerodhaClient()
        self.notifier = TelegramNotifier()
        self.ai_trader = AITradingClient()
        self.risk_manager = RiskManager()

        # State
        self.positions = {}
        self.trade_history = []
        self.is_running = False

    def connect(self, access_token: str = None) -> bool:
        """Connect to broker API."""
        logger.info("Connecting to Zerodha...")
        success = self.client.connect(access_token)

        if success:
            # Test AI connection
            try:
                test_result = self.ai_trader.analyze_market(
                    "TEST",
                    [{"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000000}] * 50,
                    100.0,
                    {}
                )
                logger.info(f"AI Client connected: {test_result['decision']}")
            except Exception as e:
                logger.warning(f"AI client test failed: {e}")

            self.notifier.send_status_update("AI Bot Started", f"Connected in {TRADING_MODE} mode")
            logger.info("Bot initialized successfully")
        else:
            self.notifier.send_error_alert("Failed to connect to Zerodha")

        return success

    def fetch_market_data(self, symbol: str, days: int = 250) -> Optional[List[Dict]]:
        """Fetch historical data for a symbol."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

        data = self.client.get_historical_data(symbol, from_date, to_date, TIMEFRAME)

        if data:
            logger.debug(f"Fetched {len(data)} candles for {symbol}")

        return data

    def scan_and_analyze(self) -> List[Dict]:
        """Scan all instruments and get AI trading decisions."""
        signals = []

        logger.info(f"Scanning {len(INSTRUMENTS)} instruments with AI...")

        # Fetch all market data first
        market_data = {}
        current_prices = {}

        for symbol in INSTRUMENTS:
            try:
                data = self.fetch_market_data(symbol)
                if data:
                    market_data[symbol] = data
                    current_prices[symbol] = data[-1]['close']
                    logger.info(f"Fetched data for {symbol}: ₹{current_prices[symbol]:.2f}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        # Send to AI for analysis
        ai_results = self.ai_trader.batch_analyze(
            INSTRUMENTS,
            market_data,
            current_prices,
            self.positions
        )

        # Process AI decisions
        for result in ai_results:
            symbol = result["symbol"]
            analysis = result["analysis"]

            if analysis["decision"] != "HOLD" and analysis["confidence"] >= 40:
                signals.append({
                    "symbol": symbol,
                    "analysis": analysis,
                    "current_price": current_prices.get(symbol, 0)
                })

        return signals

    def execute_trade(self, symbol: str, action: str, analysis: Dict, current_price: float) -> bool:
        """Execute a trade based on AI signal."""
        # Check available funds
        funds = self.client.get_funds()
        available_capital = funds.get("availablecash", MAX_POSITION_SIZE)

        logger.info(f"Available capital: ₹{available_capital:.2f}")

        # Calculate quantity
        quantity = self.risk_manager.calculate_quantity(current_price, available_capital)

        # Place order (paper or live)
        if action == "BUY":
            result = self.client.place_order(
                symbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=quantity
            )
        else:  # SELL
            result = self.client.place_order(
                symbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
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
                "confidence": analysis.get("confidence", 0)
            }

            # Send notification
            self.notifier.send_trade_alert(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=quantity,
                reason=f"AI ({analysis.get('confidence', 0)}%): {analysis.get('reasoning', '')[:50]}"
            )

            logger.info(f"Trade executed: {action} {symbol} @ ₹{current_price} x {quantity}")
            return True

        return False

    def check_positions(self):
        """Check open positions for exit signals."""
        positions_to_close = []

        # Rate limit AI calls for position checks (every ~10 minutes)
        should_get_ai_opinion = (self.scan_count % 5 == 0)

        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.client.get_ltp(symbol)
                if not current_price:
                    continue

                # Check if should close based on risk manager (stop loss / target)
                if self.risk_manager.validate_position({
                    "current_price": current_price,
                    "entry_price": position["entry_price"],
                    "type": position["type"]
                }):
                    positions_to_close.append((symbol, position, current_price))
                    continue

                # Also get AI opinion (rate limited)
                if should_get_ai_opinion:
                    data = self.fetch_market_data(symbol)
                    if data:
                        ai_analysis = self.ai_trader.analyze_market(
                        symbol,
                        data,
                        current_price,
                        self.positions
                    )

                    # If AI says SELL with reasonable confidence, close position
                    if ai_analysis["decision"] == "SELL" and ai_analysis["confidence"] >= 50:
                        positions_to_close.append((symbol, position, current_price))
                        logger.info(f"AI exit signal for {symbol}: {ai_analysis['reasoning']}")

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

        # Close positions
        for symbol, position, current_price in positions_to_close:
            action = "SELL" if position["type"] == "LONG" else "BUY"

            result = self.client.place_order(
                symbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL if position["type"] == "LONG" else kite.TRANSACTION_TYPE_BUY,
                quantity=position["quantity"]
            )

            if result:
                # Calculate PnL
                pnl = (current_price - position["entry_price"]) * position["quantity"]
                if position["type"] == "SHORT":
                    pnl = -pnl

                # Record trade
                self.trade_history.append({
                    "symbol": symbol,
                    "action": position["type"],
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "exit_time": datetime.now().isoformat(),
                    "ai_reasoning": position.get("ai_reasoning", "")
                })

                # Notify
                self.notifier.send_trade_alert(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=position["quantity"],
                    reason=f"Exit | PnL: ₹{pnl:.2f} | AI: {position.get('ai_reasoning', '')[:30]}"
                )

                # Remove from positions
                del self.positions[symbol]
                logger.info(f"Position closed: {symbol} | PnL: ₹{pnl:.2f}")

    def run_market_scan(self):
        """Run the complete market scan and AI trading logic."""
        if not self.is_running:
            logger.info("Bot not connected, skipping scan")
            return

        # Track scan count for rate limiting
        self.scan_count = getattr(self, 'scan_count', 0) + 1
        self.last_signal_scan = getattr(self, 'last_signal_scan', 0)

        logger.info(f"=== Starting Market Scan #{self.scan_count} ===")

        # First: Always check existing positions (every scan)
        self.check_positions()

        # Second: Scan for new signals every 5 scans (~10 minutes) to save API calls
        # Full market scan is expensive, so we don't do it every 2 minutes
        should_scan_signals = (self.scan_count - self.last_signal_scan) >= 5

        if should_scan_signals:
            signals = self.scan_and_analyze()
            self.last_signal_scan = self.scan_count

            for signal_data in signals:
                symbol = signal_data["symbol"]
                action = signal_data["analysis"]["decision"]
                analysis = signal_data["analysis"]
                current_price = signal_data["current_price"]

                # Skip if already in position
                if symbol in self.positions:
                    logger.info(f"Skipping {symbol} - already in position")
                    continue

                # Execute trade
                self.execute_trade(symbol, action, analysis, current_price)
        else:
            # Quick price update check
            logger.debug("Skipping full signal scan (too soon)")

        logger.info(f"=== Market Scan #{self.scan_count} Complete ===")

    def save_state(self):
        """Save bot state to file."""
        state = {
            "positions": self.positions,
            "trade_history": self.trade_history,
            "last_update": datetime.now().isoformat()
        }

        with open("ai_bot_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.debug("State saved")

    def load_state(self):
        """Load bot state from file."""
        if os.path.exists("ai_bot_state.json"):
            with open("ai_bot_state.json", "r") as f:
                state = json.load(f)
                self.positions = state.get("positions", {})
                self.trade_history = state.get("trade_history", [])
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trade_history)} trades")

    def start(self):
        """Start the AI trading bot."""
        logger.info("Starting AI Trading Bot...")

        # Load previous state
        self.load_state()

        # Connect to broker
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not self.connect(access_token):
            logger.error("Failed to connect, exiting")
            return False

        self.is_running = True

        # Schedule: continuous scanning every 2 minutes during market hours
        # Market hours: 9:15 AM to 3:35 PM IST
        schedule.every(2).minutes.do(self.run_market_scan)

        # Run on start
        time.sleep(2)
        self.run_market_scan()

        logger.info("AI Bot started. Running continuous scans from 9:15 AM to 3:35 PM...")

        # Main loop - continuously runs during market hours
        try:
            while True:
                schedule.run_pending()
                self.save_state()

                # Check if we're in market hours
                now = datetime.now()
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=35, second=0, microsecond=0)

                # If in market hours, scan every 2 minutes
                # If outside, sleep longer to reduce API calls
                if market_open <= now <= market_close:
                    time.sleep(120)  # 2 minutes during market
                else:
                    logger.info("Outside market hours, waiting...")
                    time.sleep(300)  # 5 minutes outside market

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.notifier.send_status_update("Bot Stopped", "User requested shutdown")
        except Exception as e:
            logger.error(f"Bot error: {e}")
            self.notifier.send_error_alert(f"Bot error: {e}")

        return True

    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        self.save_state()
        logger.info("Bot stopped")


if __name__ == "__main__":
    bot = AITradingBot()
    bot.start()