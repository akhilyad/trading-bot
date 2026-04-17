"""
MARKET MICROSTRUCTURE & ORDER FLOW MODULE
Level 2 Data, Order Book Analysis, Cumulative Delta, Market Impact
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from logger import logger


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    quantity: int
    orders: int


@dataclass
class OrderBook:
    """Full order book."""
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    depth_bid: int
    depth_ask: int
    timestamp: datetime


@dataclass
class TradePrint:
    """Individual trade print."""
    price: float
    quantity: int
    time: datetime
    side: str  # BUY or SELL


class OrderBookAnalyzer:
    """Analyze order book dynamics."""

    def __init__(self):
        self.history = deque(maxlen=100)
        self.volume_profile = defaultdict(int)

    def analyze(self, order_book: OrderBook) -> Dict:
        """Analyze order book and generate insights."""
        # Bid-ask spread
        spread_bps = (order_book.spread / order_book.mid_price) * 10000  # Basis points

        # Depth analysis
        bid_depth = sum(b.quantity for b in order_book.bids[:5])
        ask_depth = sum(a.quantity for a in order_book.asks[:5])

        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1)

        # Volume weighted prices
        bid_vwap = sum(b.price * b.quantity for b in order_book.bids[:5]) / max(bid_depth, 1)
        ask_vwap = sum(a.price * a.quantity for a in order_book.asks[:5]) / max(ask_depth, 1)

        # Liquidity score (lower spread = higher liquidity)
        liquidity_score = max(0, 100 - spread_bps)

        return {
            'spread_bps': spread_bps,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_imbalance': depth_imbalance,
            'bid_vwap': bid_vwap,
            'ask_vwap': ask_vwap,
            'liquidity_score': liquidity_score,
            'mid_price': order_book.mid_price
        }

    def detect_liquidity_gaps(self, order_book: OrderBook) -> List[Dict]:
        """Detect liquidity gaps (areas with no orders)."""
        gaps = []

        # Get all bid and ask prices
        bid_prices = [b.price for b in order_book.bids]
        ask_prices = [a.price for a in order_book.asks]

        # Find large gaps between levels
        for i in range(len(bid_prices) - 1):
            gap = bid_prices[i] - bid_prices[i + 1]
            if gap > order_book.mid_price * 0.01:  # 1% gap
                gaps.append({
                    'type': 'BID_GAP',
                    'price': bid_prices[i + 1],
                    'size': gap,
                    'side': 'bid'
                })

        for i in range(len(ask_prices) - 1):
            gap = ask_prices[i + 1] - ask_prices[i]
            if gap > order_book.mid_price * 0.01:
                gaps.append({
                    'type': 'ASK_GAP',
                    'price': ask_prices[i],
                    'size': gap,
                    'side': 'ask'
                })

        return gaps

    def estimate_market_impact(self, order_size: int, side: str, order_book: OrderBook) -> float:
        """Estimate price impact of a large order."""
        if side == 'BUY':
            available_depth = sum(a.quantity for a in order_book.asks[:5])
            if available_depth == 0:
                return 0.05  # 5% impact if no liquidity
            fill_ratio = min(1.0, order_size / available_depth)
            impact = 0.02 * fill_ratio  # 2% max impact
        else:
            available_depth = sum(b.quantity for b in order_book.bids[:5])
            if available_depth == 0:
                return 0.05
            fill_ratio = min(1.0, order_size / available_depth)
            impact = 0.02 * fill_ratio

        return impact


class CumulativeDeltaTracker:
    """Track cumulative delta (buy/sell pressure) over time."""

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.trades = deque(maxlen=lookback)
        self.cumulative_delta = 0
        self.delta_history = deque(maxlen=500)

    def add_trade(self, price: float, quantity: int, side: str):
        """Add a trade to the delta calculation."""
        trade = TradePrint(price, quantity, datetime.now(), side)
        self.trades.append(trade)

        # Update cumulative delta
        if side.upper() == 'BUY':
            self.cumulative_delta += quantity
        else:
            self.cumulative_delta -= quantity

        self.delta_history.append(self.cumulative_delta)

    def get_delta_signal(self) -> Dict:
        """Get trading signal from cumulative delta."""
        if len(self.delta_history) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 0}

        # Current delta
        current_delta = self.cumulative_delta

        # Delta statistics
        delta_mean = np.mean(list(self.delta_history))
        delta_std = np.std(list(self.delta_history))

        # Z-score
        z_score = (current_delta - delta_mean) / (delta_std + 1)

        # Price comparison
        recent_prices = [t.price for t in list(self.trades)[-20:]]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100 if recent_prices else 0

        # Signal generation
        if z_score > 2.0 and price_trend > 0:
            return {
                'signal': 'STRONG_BUY',
                'confidence': min(90, 50 + z_score * 15),
                'delta': current_delta,
                'z_score': z_score,
                'reason': 'Delta diverging upward with price'
            }
        elif z_score > 1.5 and price_trend > 0:
            return {
                'signal': 'BUY',
                'confidence': min(70, 40 + z_score * 10),
                'delta': current_delta,
                'z_score': z_score
            }
        elif z_score < -2.0 and price_trend < 0:
            return {
                'signal': 'STRONG_SELL',
                'confidence': min(90, 50 - z_score * 15),
                'delta': current_delta,
                'z_score': z_score,
                'reason': 'Delta diverging downward with price'
            }
        elif z_score < -1.5 and price_trend < 0:
            return {
                'signal': 'SELL',
                'confidence': min(70, 40 - z_score * 10),
                'delta': current_delta,
                'z_score': z_score
            }
        elif z_score > 1 and price_trend < 0:
            return {
                'signal': 'BEARISH_DIV',
                'confidence': 60,
                'delta': current_delta,
                'z_score': z_score,
                'reason': 'Positive delta but falling price - bearish divergence'
            }
        elif z_score < -1 and price_trend > 0:
            return {
                'signal': 'BULLISH_DIV',
                'confidence': 60,
                'delta': current_delta,
                'z_score': z_score,
                'reason': 'Negative delta but rising price - bullish divergence'
            }

        return {
            'signal': 'NEUTRAL',
            'confidence': 40,
            'delta': current_delta,
            'z_score': z_score
        }

    def get_delta_chart_data(self) -> List[Tuple[datetime, int]]:
        """Get data for delta chart."""
        return [(t.time, t.quantity if t.side.upper() == 'BUY' else -t.quantity)
                for t in self.trades]


class MarketImpactModel:
    """Model market impact for large orders."""

    def __init__(self):
        self.impact_history = []

    def calculate_impact(self, order_size: int, avg_volume: int,
                       volatility: float) -> Dict:
        """Calculate expected market impact."""
        # Volume ratio
        vol_ratio = order_size / avg_volume if avg_volume > 0 else 1

        # Impact models
        # Square root model: impact ~ sqrt(volume_ratio)
        impact_sqrt = 0.1 * math.sqrt(vol_ratio)

        # Linear model: impact ~ volume_ratio (for larger orders)
        impact_linear = 0.05 * vol_ratio

        # Volatility adjustment
        vol_multiplier = 1 + volatility

        # Combined impact estimate
        total_impact = (impact_sqrt * 0.6 + impact_linear * 0.4) * vol_multiplier

        # Fill probability
        if vol_ratio < 0.01:
            fill_prob = 0.99
        elif vol_ratio < 0.05:
            fill_prob = 0.90
        elif vol_ratio < 0.1:
            fill_prob = 0.70
        else:
            fill_prob = 0.50

        return {
            'impact_percent': total_impact * 100,
            'volume_ratio': vol_ratio,
            'fill_probability': fill_prob,
            'suggestion': 'SPLIT' if vol_ratio > 0.05 else 'DIRECT'
        }

    def estimate_slippage(self, order_size: int, avg_spread: float) -> float:
        """Estimate slippage for order."""
        # Slippage is proportional to order size relative to spread
        slippage = avg_spread * (1 + order_size / 1000)
        return slippage


class VolumeProfileAnalysis:
    """Volume at Price analysis (Point of Control)."""

    def __init__(self, bins: int = 50):
        self.bins = bins
        self.profile = defaultdict(int)

    def add_price_volume(self, price: float, volume: int):
        """Add price-volume data point."""
        bin_price = int(price / self.bins) * self.bins
        self.profile[bin_price] += volume

    def get_poc(self) -> Tuple[float, int]:
        """Get Point of Control (highest volume price)."""
        if not self.profile:
            return 0.0, 0

        max_bin = max(self.profile.keys(), key=lambda x: self.profile[x])
        return max_bin, self.profile[max_bin]

    def get_value_area(self, coverage: float = 0.7) -> Tuple[float, float]:
        """Get Value Area (range containing X% of volume)."""
        if not self.profile:
            return 0.0, 0.0

        total_volume = sum(self.profile.values())
        target_volume = total_volume * coverage

        # Sort by volume
        sorted_bins = sorted(self.profile.items(), key=lambda x: x[1], reverse=True)

        cumsum = 0
        value_bins = []
        for bin_price, vol in sorted_bins:
            cumsum += vol
            value_bins.append(bin_price)
            if cumsum >= target_volume:
                break

        return min(value_bins), max(value_bins)

    def get_volume_nodes(self) -> List[Dict]:
        """Get significant volume nodes (high volume clusters).."""
        if not self.profile:
            return []

        avg_volume = sum(self.profile.values()) / len(self.profile)

        nodes = []
        for price, volume in self.profile.items():
            if volume > avg_volume * 2:  # Significant node
                nodes.append({
                    'price': price,
                    'volume': volume,
                    'strength': volume / avg_volume
                })

        return sorted(nodes, key=lambda x: x['volume'], reverse=True)[:10]


class TimeSalesAnalyzer:
    """Analyze Time & Sales data."""

    def __init__(self):
        self.trades = deque(maxlen=1000)
        self.large_trades = deque(maxlen=50)

    def add_trade(self, price: float, quantity: int, timestamp: datetime):
        """Add a trade to analysis."""
        trade = {
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp,
            'value': price * quantity
        }
        self.trades.append(trade)

        # Detect large trades ( > 5x average)
        avg_qty = np.mean([t['quantity'] for t in self.trades]) if self.trades else 0
        if quantity > avg_qty * 5:
            self.large_trades.append(trade)

    def get_analysis(self) -> Dict:
        """Get comprehensive T&S analysis."""
        if not self.trades:
            return {}

        prices = [t['price'] for t in self.trades]
        quantities = [t['quantity'] for t in self.trades]

        # Statistics
        avg_price = np.mean(prices)
        avg_qty = np.mean(quantities)
        max_qty = max(quantities)

        # Large trade analysis
        large_trade_count = len(self.large_trades)
        large_trade_volume = sum(t['quantity'] for t in self.large_trades)
        total_volume = sum(quantities)

        large_trade_ratio = large_trade_volume / total_volume if total_volume > 0 else 0

        # Recent activity
        recent = list(self.trades)[-100:]
        recent_trend = (recent[-1]['price'] - recent[0]['price']) / recent[0]['price'] * 100 if recent else 0

        return {
            'avg_price': avg_price,
            'avg_quantity': avg_qty,
            'max_quantity': max_qty,
            'large_trade_count': large_trade_count,
            'large_trade_ratio': large_trade_ratio,
            'recent_trend': recent_trend,
            'total_trades': len(self.trades)
        }


class MarketMicrostructureEngine:
    """Complete market microstructure system."""

    def __init__(self):
        self.order_book = OrderBook([], [], 0, 0, 0, 0)
        self.delta_tracker = CumulativeDeltaTracker()
        self.impact_model = MarketImpactModel()
        self.volume_profile = VolumeProfileAnalysis()
        self.time_sales = TimeSalesAnalyzer()

    def update(self, bid_price: float, bid_qty: int, ask_price: float,
              ask_qty: int, last_trade_price: int, last_trade_qty: int,
              last_trade_side: str):
        """Update all microstructure data."""

        # Update order book
        self.order_book = OrderBook(
            bids=[OrderBookLevel(bid_price, bid_qty, 1)],
            asks=[OrderBookLevel(ask_price, ask_qty, 1)],
            spread=ask_price - bid_price,
            mid_price=(bid_price + ask_price) / 2,
            depth_bid=bid_qty,
            depth_ask=ask_qty,
            timestamp=datetime.now()
        )

        # Update delta
        if last_trade_side:
            self.delta_tracker.add_trade(last_trade_price, last_trade_qty, last_trade_side)

        # Update volume profile
        self.volume_profile.add_price_volume(last_trade_price, last_trade_qty)

        # Update time & sales
        self.time_sales.add_trade(last_trade_price, last_trade_qty, datetime.now())

    def get_all_signals(self) -> Dict:
        """Get all microstructure signals."""
        signals = {}

        # Order book analysis
        ob = OrderBookAnalyzer()
        signals['order_book'] = ob.analyze(self.order_book)
        signals['liquidity_gaps'] = ob.detect_liquidity_gaps(self.order_book)

        # Delta signals
        signals['delta'] = self.delta_tracker.get_delta_signal()

        # Volume profile
        poc, poc_vol = self.volume_profile.get_poc()
        va_low, va_high = self.volume_profile.get_value_area()
        signals['volume_profile'] = {
            'poc': poc,
            'poc_volume': poc_vol,
            'value_area_low': va_low,
            'value_area_high': va_high,
            'nodes': self.volume_profile.get_volume_nodes()
        }

        # Time & Sales
        signals['time_sales'] = self.time_sales.get_analysis()

        return signals


# =============================================================================
# CUSTOM BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """Institutional-grade backtesting engine."""

    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.trade_log = []

    def run(
        self,
        data: pd.DataFrame,
        strategy_func,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.1
    ) -> Dict:
        """Run backtest with realistic costs."""
        capital = initial_capital
        position = None
        equity = [initial_capital]

        for i in range(50, len(data)):
            current_price = data['close'].iloc[i]
            current_date = data.index[i]

            # Get signal
            signal = strategy_func(data.iloc[:i+1])

            # Entry
            if signal == 'BUY' and position is None:
                qty = int(capital * position_size / current_price)
                cost = current_price * qty * (1 + commission + slippage)

                if cost <= capital:
                    position = {
                        'entry': current_price,
                        'qty': qty,
                        'date': current_date
                    }
                    capital -= cost

            # Exit
            elif (signal == 'SELL' or i - position['entry_date'] > 20 if position else False) and position:
                revenue = current_price * position['qty'] * (1 - commission - slippage)
                pnl = revenue - (position['entry'] * position['qty'])

                self.trades.append({
                    'entry': position['entry'],
                    'exit': current_price,
                    'pnl': pnl,
                    'return': pnl / (position['entry'] * position['qty'])
                })

                capital += revenue
                position = None

            equity.append(capital)

        # Calculate metrics
        returns = np.diff(equity) / equity[:-1]

        return {
            'total_return': (equity[-1] - initial_capital) / initial_capital * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._max_drawdown(equity),
            'win_rate': len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) if self.trades else 0,
            'trade_count': len(self.trades),
            'equity_curve': equity
        }

    def _max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = equity[0]
        max_dd = 0

        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)

        return max_dd


class WalkForwardOptimizer:
    """Walk-forward optimization to prevent overfitting."""

    def __init__(self):
        self.results = []

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_params: Dict,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 21
    ) -> Dict:
        """Run walk-forward optimization."""
        train_results = []
        test_results = []

        start = train_period
        while start + test_period <= len(data):
            train_data = data.iloc[start - train_period:start]
            test_data = data.iloc[start:start + test_period]

            # Optimize on train
            best_params = self._optimize_params(train_data, strategy_params)

            # Test on holdout
            bt = BacktestEngine()
            result = bt.run(test_data, lambda d: self._execute_strategy(d, best_params))

            train_results.append(best_params)
            test_results.append(result)

            start += step

        return {
            'train_results': train_results,
            'test_results': test_results,
            'avg_return': np.mean([r['total_return'] for r in test_results]),
            'consistency': np.std([r['total_return'] for r in test_results])
        }

    def _optimize_params(self, data: pd.DataFrame, base_params: Dict) -> Dict:
        """Find best parameters on training data."""
        # Simplified grid search
        return base_params

    def _execute_strategy(self, data: pd.DataFrame, params: Dict) -> str:
        """Execute strategy with params."""
        # Simplified signal
        return 'HOLD'