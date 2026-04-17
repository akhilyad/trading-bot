"""
Telegram Notification Module
Sends trade alerts and status updates to Telegram.
"""

import os
from typing import Optional
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from logger import logger


class TelegramNotifier:
    """Handles sending messages to Telegram."""

    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            logger.info(f"Telegram message sent: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_trade_alert(self, symbol: str, action: str, price: float, quantity: int, reason: str = "") -> bool:
        """Send a trade alert message."""
        emoji = "🟢" if action.upper() == "BUY" else "🔴" if action.upper() == "SELL" else "⚪"
        message = f"""
{emoji} *TRADE ALERT*

*Symbol:* {symbol}
*Action:* {action.upper()}
*Price:* ₹{price:.2f}
*Quantity:* {quantity}
*Reason:* {reason}
"""
        return self.send_message(message)

    def send_error_alert(self, error_message: str) -> bool:
        """Send an error alert."""
        message = f"🚨 *ERROR*\n\n{error_message}"
        return self.send_message(message)

    def send_status_update(self, status: str, details: str = "") -> bool:
        """Send a status update."""
        message = f"📊 *STATUS*\n\n*{status}*\n{details}"
        return self.send_message(message)

    def send_daily_summary(self, trades: list, pnl: float) -> bool:
        """Send daily trading summary."""
        trade_list = "\n".join([f"- {t['symbol']}: {t['action']} @ ₹{t['price']}" for t in trades])
        message = f"""
📈 *DAILY SUMMARY*

*Trades:* {len(trades)}
*PnL:* ₹{pnl:.2f}

{trade_list}
"""
        return self.send_message(message)