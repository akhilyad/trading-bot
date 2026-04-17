"""
Main Trading Bot
Orchestrates all components: API, Strategy, Notifications.
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
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, TRADING_MODE
)
import kiteconnect as kite
from zerodha_client import ZerodhaClient
from telegram_notifier import TelegramNotifier
from strategy import MACrossoverStrategy, RiskManager
from logger import logger

# Load environment variables
load_dotenv()


class TradingBot:
    """Main trading bot class."""

    def __init__(self):
        self.client = ZerodhaClient()
        self.notifier = TelegramNotifier()
        self.strategy = MACrossoverStrategy()
        self.risk_manager = RiskManager()

        # State
        self.positions = {}  # symbol -> position data
        self.trade_history = []
        self.is_running = False

    def connect(self, access_token: str = None) -> bool:
        """Connect to broker API."""
        logger.info("Connecting to Zerodha...")
        success = self.client.connect(access_token)

        if success:
            self.notifier.send_status_update("Bot Started", f"Connected in {TRADING_MODE} mode")
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
        """Scan all instruments and generate trade signals."""
        signals = []

        logger.info(f"Scanning {len(INSTRUMENTS)} instruments...")

        for symbol in INSTRUMENTS:
            try:
                # Fetch data
                data = self.fetch_market_data(symbol)
                if not data:
                    continue

                # Analyze
                analysis = self.strategy.analyze(data)

                # Log signal
                if analysis["signal"] != "HOLD":
                    logger.info(f"Signal: {symbol} -> {analysis['signal']}")
                    signals.append({
                        "symbol": symbol,
                        "analysis": analysis
                    })

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        return signals

    def execute_trade(self, symbol: str, action: str, analysis: Dict) -> bool:
        """Execute a trade based on signal."""
        current_price = analysis["current_price"]

        # Check available funds
        funds = self.client.get_funds()
        available_capital = funds.get("availablecash", 0)

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
                "type": action,
                "entry_price": current_price,
                "quantity": quantity,
                "entry_time": datetime.now().isoformat(),
                "stop_loss": analysis.get("stop_loss"),
                "target": analysis.get("target")
            }

            # Send notification
            self.notifier.send_trade_alert(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=quantity,
                reason=analysis["reason"]
            )

            logger.info(f"Trade executed: {action} {symbol} @ ₹{current_price} x {quantity}")
            return True

        return False

    def check_positions(self):
        """Check open positions for exit signals."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            try:
                current_price = self.client.get_ltp(symbol)
                if not current_price:
                    continue

                # Check if should close
                if self.risk_manager.validate_position({
                    "current_price": current_price,
                    "entry_price": position["entry_price"],
                    "type": position["type"]
                }):
                    positions_to_close.append((symbol, position, current_price))

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
                    "exit_time": datetime.now().isoformat()
                })

                # Notify
                self.notifier.send_trade_alert(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=position["quantity"],
                    reason=f"Exit | PnL: ₹{pnl:.2f}"
                )

                # Remove from positions
                del self.positions[symbol]
                logger.info(f"Position closed: {symbol} | PnL: ₹{pnl:.2f}")

    def run_market_scan(self):
        """Run the complete market scan and trading logic."""
        if not self.is_running:
            logger.info("Bot not connected, skipping scan")
            return

        logger.info("=== Starting Market Scan ===")

        # First check existing positions
        self.check_positions()

        # Scan for new signals (only if not in paper mode or if allowed)
        signals = self.scan_and_analyze()

        for signal_data in signals:
            symbol = signal_data["symbol"]
            action = signal_data["analysis"]["signal"]
            analysis = signal_data["analysis"]

            # Skip if already in position
            if symbol in self.positions:
                logger.info(f"Skipping {symbol} - already in position")
                continue

            # Execute trade
            self.execute_trade(symbol, action, analysis)

        logger.info("=== Market Scan Complete ===")

    def save_state(self):
        """Save bot state to file."""
        state = {
            "positions": self.positions,
            "trade_history": self.trade_history,
            "last_update": datetime.now().isoformat()
        }

        with open("bot_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.debug("State saved")

    def load_state(self):
        """Load bot state from file."""
        if os.path.exists("bot_state.json"):
            with open("bot_state.json", "r") as f:
                state = json.load(f)
                self.positions = state.get("positions", {})
                self.trade_history = state.get("trade_history", [])
                logger.info(f"Loaded state: {len(self.positions)} positions, {len(self.trade_history)} trades")

    def start(self):
        """Start the trading bot."""
        logger.info("Starting Trading Bot...")

        # Load previous state
        self.load_state()

        # Connect to broker
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not self.connect(access_token):
            logger.error("Failed to connect, exiting")
            return False

        self.is_running = True

        # Schedule market scans
        # Scan at market open and every hour during market hours
        schedule.every().day.at("09:15").do(self.run_market_scan)
        schedule.every().day.at("10:00").do(self.run_market_scan)
        schedule.every().day.at("11:00").do(self.run_market_scan)
        schedule.every().day.at("12:00").do(self.run_market_scan)
        schedule.every().day.at("13:00").do(self.run_market_scan)
        schedule.every().day.at("14:00").do(self.run_market_scan)
        schedule.every().day.at("15:00").do(self.run_market_scan)

        # Also run on bot start
        time.sleep(2)
        self.run_market_scan()

        logger.info("Bot started. Running scheduled scans...")

        # Main loop
        try:
            while True:
                schedule.run_pending()
                self.save_state()
                time.sleep(60)  # Check every minute

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
    bot = TradingBot()
    bot.start()