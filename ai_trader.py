"""
AI Trading Module
Uses Claude/Minimax API to analyze markets and generate trading signals.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from logger import logger

load_dotenv()


class AITradingClient:
    """Client for AI-powered trading decisions."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None
    ):
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", base_url or "https://opencode.ai/zen")
        self.api_key = os.getenv("ANTHROPIC_AUTH_TOKEN", api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", model or "minimax-m2.5-free")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_market(
        self,
        symbol: str,
        historical_data: List[Dict],
        current_price: float,
        positions: Dict = None
    ) -> Dict:
        """
        Send market data to AI and get trading decision.

        Args:
            symbol: Stock symbol
            historical_data: OHLC data
            current_price: Current market price
            positions: Current open positions

        Returns:
            Dict with keys: decision (BUY/SELL/HOLD), reasoning, confidence
        """
        # Prepare data for AI
        df = pd.DataFrame(historical_data)
        df['close'] = pd.to_numeric(df['close'])

        # Calculate indicators
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean() if len(df) >= 200 else None

        # Recent price action
        recent_5 = df['close'].tail(5).tolist()
        recent_20 = df['close'].tail(20).tolist()

        # Volume trend
        volume = df['volume'].tail(20).mean() if 'volume' in df.columns else 0

        # Price momentum
        if len(df) >= 14:
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            rsi = df['rsi'].iloc[-1]
        else:
            rsi = 50

        # Current positions for this symbol
        current_position = positions.get(symbol) if positions else None

        # Build prompt
        prompt = f"""You are an expert stock trading analyst. Analyze {symbol} and give a trading decision.

CURRENT DATA:
- Price: ₹{current_price:.2f}
- 20-Day MA: ₹{df['ma20'].iloc[-1]:.2f} (if available)
- 50-Day MA: ₹{df['ma50'].iloc[-1]:.2f} (if available)
- 14-Day RSI: {rsi:.2f}
- Avg Volume (20 days): {volume:,.0f}

RECENT PRICES (last 5 days): {recent_5}

CURRENT POSITION: {"Long" if current_position and current_position.get('type') == 'LONG' else "Short" if current_position and current_position.get('type') == 'SHORT' else "None"}

Based on technical analysis and market conditions, decide:
- BUY: Strong bullish signal
- SELL: Close position or short
- HOLD: No clear signal

Respond in this exact JSON format:
{{"decision": "BUY/SELL/HOLD", "reasoning": "2-3 sentence explanation", "confidence": 0-100}}"""

        try:
            response = self._call_api(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"decision": "HOLD", "reasoning": f"AI error: {str(e)}", "confidence": 0}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _call_api(self, prompt: str) -> str:
        """Call the AI API."""
        # Anthropic-compatible API format
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }

        # Try different endpoint formats
        endpoints = [
            f"{self.base_url}/v1/chat/completions",
            f"{self.base_url}/v1/messages",
            f"{self.base_url}/chat/completions"
        ]

        for endpoint in endpoints:
            try:
                response = requests.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle different response formats
                    if 'choices' in data:
                        return data['choices'][0]['message']['content']
                    elif 'content' in data:
                        return data['content'][0]['text'] if isinstance(data['content'], list) else data['content']
                    else:
                        return str(data)

            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue

        raise Exception("All API endpoints failed")

    def _parse_response(self, response: str) -> Dict:
        """Parse AI response into structured format."""
        if not response:
            return {"decision": "HOLD", "reasoning": "No response from AI", "confidence": 0}

        try:
            # Try to extract JSON from response
            import re

            # Look for JSON block
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "decision": data.get("decision", "HOLD").upper(),
                    "reasoning": data.get("reasoning", "No explanation")[:200],
                    "confidence": int(data.get("confidence", 50))
                }

            # Try to extract any JSON
            json_match = re.search(r'\{.+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "decision": data.get("decision", "HOLD").upper(),
                    "reasoning": data.get("reasoning", "No explanation")[:200],
                    "confidence": int(data.get("confidence", 50))
                }

            # Fallback: look for keywords in text
            response_upper = response.upper()
            if "BUY" in response_upper and "BUY" not in response_upper[:50]:
                return {"decision": "BUY", "reasoning": response[:200], "confidence": 60}
            elif "SELL" in response_upper and "SELL" not in response_upper[:50]:
                return {"decision": "SELL", "reasoning": response[:200], "confidence": 60}
            else:
                return {"decision": "HOLD", "reasoning": response[:200], "confidence": 30}

        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return {"decision": "HOLD", "reasoning": "Parse error", "confidence": 0}

    def batch_analyze(
        self,
        symbols: List[str],
        market_data: Dict[str, List[Dict]],
        current_prices: Dict[str, float],
        positions: Dict
    ) -> List[Dict]:
        """Analyze multiple symbols."""
        results = []

        for symbol in symbols:
            data = market_data.get(symbol, [])
            price = current_prices.get(symbol, 0)

            if not data or not price:
                continue

            analysis = self.analyze_market(symbol, data, price, positions)

            results.append({
                "symbol": symbol,
                "analysis": analysis
            })

            logger.info(f"AI: {symbol} -> {analysis['decision']} ({analysis['confidence']}% confidence)")

            # Rate limiting between calls
            import time
            time.sleep(1)

        return results


# Test function
if __name__ == "__main__":
    client = AITradingClient()

    # Test with sample data
    sample_data = [
        {"date": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000000},
        {"date": "2024-01-02", "open": 103, "high": 108, "low": 102, "close": 106, "volume": 1100000},
    ] * 100  # Repeat for enough data

    result = client.analyze_market("RELIANCE", sample_data, 1500.00, {})
    print(f"Result: {result}")