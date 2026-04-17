"""
Zerodha Kite Connect API Wrapper
Handles authentication, data fetching, and order placement.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import kiteconnect as kite
from config import (
    KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN,
    TRADING_MODE, INSTRUMENTS
)
from logger import logger


class ZerodhaClient:
    """Wrapper for Zerodha Kite Connect API."""

    def __init__(self):
        self.kite = None
        self.access_token = None
        self.is_connected = False

    def connect(self, access_token: str = None) -> bool:
        """Initialize Kite Connect with access token."""
        try:
            self.access_token = access_token or KITE_ACCESS_TOKEN
            self.kite = kite.KiteConnect(api_key=KITE_API_KEY)

            if self.access_token:
                self.kite.set_access_token(self.access_token)

            # Test connection with profile
            profile = self.kite.profile()
            logger.info(f"Connected to Zerodha: {profile.get('user_name', 'Unknown')}")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            return False

    def generate_access_token(self, request_token: str) -> Optional[str]:
        """Generate access token from request token."""
        try:
            data = self.kite.generate_session(
                request_token,
                api_secret=KITE_API_SECRET
            )
            access_token = data.get("access_token")
            logger.info("Access token generated successfully")
            return access_token
        except Exception as e:
            logger.error(f"Failed to generate access token: {e}")
            return None

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """Get instrument token for a symbol."""
        try:
            instruments = self.kite.instruments(exchange=exchange)
            for inst in instruments:
                if inst.get("tradingsymbol") == symbol:
                    return inst.get("instrument_token")
            return None
        except Exception as e:
            logger.error(f"Failed to get instrument token for {symbol}: {e}")
            return None

    def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        """Get quote for multiple instruments."""
        try:
            quotes = self.kite.quote(["NSE:" + s for s in symbols])
            return quotes
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}

    def get_historical_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "day"
    ) -> Optional[List[Dict]]:
        """Fetch historical candle data."""
        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return None

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return data
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price for a symbol."""
        try:
            quotes = self.kite.quote(["NSE:" + symbol])
            return quotes.get("NSE:" + symbol, {}).get("last_price")
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        product: str = "CNC",
        order_type: str = "MARKET"
    ) -> Optional[Dict]:
        """Place a buy/sell order."""
        try:
            # Don't place real orders in paper mode
            if TRADING_MODE == "paper":
                logger.info(f"[PAPER] Would place {transaction_type} order: {symbol} x {quantity}")
                return {"order_id": "PAPER_ORDER", "status": "success"}

            # Live order
            order_id = self.kite.place_order(
                variety="regular",
                exchange="NSE",
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type
            )
            logger.info(f"Order placed: {transaction_type} {symbol} x {quantity}, Order ID: {order_id}")
            return {"order_id": order_id, "status": "success"}
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        try:
            positions = self.kite.positions()
            return positions.get("day", []) + positions.get("net", [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_orders(self) -> List[Dict]:
        """Get today's orders."""
        try:
            orders = self.kite.orders()
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def get_funds(self) -> Dict[str, float]:
        """Get available funds."""
        try:
            margins = self.kite.margins()
            return margins.get("equity", {})
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {}

    def logout(self):
        """Logout and revoke access token."""
        try:
            self.kite.logout()
            logger.info("Logged out from Zerodha")
        except Exception as e:
            logger.error(f"Logout error: {e}")