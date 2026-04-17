"""
Configuration settings for the trading bot.
Load from environment variables for security.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Zerodha Kite API Credentials
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Trading Settings
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"
INSTRUMENTS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]  # NSE stocks to scan
TIMEFRAME = "day"  # daily candles

# Strategy Parameters
SHORT_MA_PERIOD = 50
LONG_MA_PERIOD = 200

# Risk Management
MAX_POSITION_SIZE = 100000  # Max rupees per position (paper trading simulation)
STOP_LOSS_PERCENT = 2.0   # 2% stop loss
TARGET_PROFIT_PERCENT = 4.0  # 4% target

# Trading Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Logging
LOG_FILE = "trading_bot.log"
LOG_LEVEL = "INFO"