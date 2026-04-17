"""
EXECUTION ALGORITHMS MODULE
Professional order execution: VWAP, TWAP, Implementation Shortfall, Iceberg
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from logger import logger


@dataclass
class OrderParams:
    """Order parameters for execution algorithms."""
    symbol: str
    quantity: int
    side: str  # BUY or SELL
    limit_price: Optional[float] = None
    algo: str = "MARKET"  # MARKET, VWAP, TWAP, IS, ICEBERG
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_id: str
    status: str  # FILLED, PARTIAL, CANCELLED
    filled_quantity: int
    avg_fill_price: float
    total_value: float
    slippage: float  # Actual vs expected
    execution_time: float


class TWAPExecutor:
    """Time-Weighted Average Price execution."""

    def __init__(self, client):
        self.client = client
        self.orders = []

    def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute order using TWAP."""
        if not params.start_time:
            params.start_time = datetime.now()
        if not params.end_time:
            params.end_time = params.start_time + timedelta(minutes=30)

        # Calculate number of slices
        duration = (params.end_time - params.start_time).total_seconds()
        slice_duration = 60  # 1 minute slices
        num_slices = max(1, int(duration / slice_duration))

        # Quantity per slice
        quantity_per_slice = params.quantity // num_slices
        remaining = params.quantity % num_slices

        total_filled = 0
        total_value = 0.0

        logger.info(f"TWAP: Executing {params.quantity} {params.symbol} in {num_slices} slices")

        for i in range(num_slices):
            slice_qty = quantity_per_slice + (1 if i < remaining else 0)

            # Get current market price
            current_price = self.client.get_ltp(params.symbol) or params.limit_price

            if not current_price:
                logger.warning(f"TWAP: Could not get price for {params.symbol}")
                continue

            # Execute slice
            try:
                # In production: use actual broker API
                # For now: simulate
                filled = slice_qty
                fill_price = current_price * (1 + 0.001 if params.side == 'BUY' else -0.001)

                total_filled += filled
                total_value += filled * fill_price

                logger.debug(f"TWAP slice {i+1}/{num_slices}: {filled} @ ₹{fill_price:.2f}")

            except Exception as e:
                logger.error(f"TWAP slice failed: {e}")

            # Wait before next slice
            if i < num_slices - 1:
                time.sleep(slice_duration)

        # Calculate results
        avg_price = total_value / total_filled if total_filled > 0 else 0

        # Expected price (TWAP of market)
        expected_price = params.limit_price or avg_price
        slippage = ((avg_price - expected_price) / expected_price) * 100 if expected_price > 0 else 0

        return ExecutionResult(
            order_id=f"TWAP_{params.symbol}_{int(time.time())}",
            status="FILLED" if total_filled == params.quantity else "PARTIAL",
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_value=total_value,
            slippage=slippage,
            execution_time=duration
        )


class VWAPExecutor:
    """Volume-Weighted Average Price execution."""

    def __init__(self, client):
        self.client = client
        self.volume_history = deque(maxlen=100)

    def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute order using VWAP."""
        if not params.end_time:
            params.end_time = datetime.now() + timedelta(minutes=30)

        # Get historical volume profile
        self._update_volume_profile(params.symbol)

        # Calculate target participation rate
        avg_volume = self._get_avg_volume(params.symbol)
        target_rate = params.quantity / (avg_volume * 0.1) if avg_volume > 0 else 0.1  # 10% of 10-min volume

        total_filled = 0
        total_value = 0.0

        logger.info(f"VWAP: Executing {params.quantity} {params.symbol} at {target_rate:.2%} participation")

        # Execute in small chunks based on volume
        chunk_size = max(1, int(avg_volume * 0.02))  # 2% of 10-min volume

        while total_filled < params.quantity:
            # Get current market data
            current_price = self.client.get_ltp(params.symbol) or params.limit_price

            if not current_price:
                break

            # Calculate fill quantity based on participation
            fill_qty = min(chunk_size, params.quantity - total_filled)

            try:
                fill_price = current_price * (1 + 0.0005 if params.side == 'BUY' else -0.0005)

                total_filled += fill_qty
                total_value += fill_qty * fill_price

            except Exception as e:
                logger.error(f"VWAP chunk failed: {e}")

            time.sleep(30)  # Check every 30 seconds

        avg_price = total_value / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            order_id=f"VWAP_{params.symbol}_{int(time.time())}",
            status="FILLED" if total_filled == params.quantity else "PARTIAL",
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_value=total_value,
            slippage=0,  # Would calculate vs VWAP benchmark
            execution_time=30
        )

    def _update_volume_profile(self, symbol: str):
        """Update volume history for VWAP calculation."""
        # In production: fetch real volume data
        # Simulated
        for _ in range(100):
            self.volume_history.append(1000000)  # 10L average volume

    def _get_avg_volume(self, symbol: str) -> int:
        """Get average volume."""
        if not self.volume_history:
            return 1000000
        return sum(self.volume_history) / len(self.volume_history)


class ImplementationShortfall:
    """Implementation Shortfall (IS) algorithm - minimize slippage."""

    def __init__(self):
        self.alpha = 0.5  # Urgency parameter (0 = patient, 1 = urgent)

    def calculate_urgency(self, remaining_time: float, remaining_qty: int,
                          total_qty: int) -> float:
        """Calculate urgency based on remaining time and quantity."""
        time_elapsed = 1 - (remaining_time / 1800)  # Assuming 30 min window
        qty_filled = 1 - (remaining_qty / total_qty)

        # If behind schedule, increase urgency
        if qty_filled < time_elapsed:
            self.alpha = min(1.0, self.alpha + 0.1)

        return self.alpha

    def execute(self, params: OrderParams, market_data: Dict) -> ExecutionResult:
        """Execute using Implementation Shortfall algorithm."""
        start_time = datetime.now()
        if not params.end_time:
            params.end_time = start_time + timedelta(minutes=30)

        remaining_qty = params.quantity
        total_value = 0.0

        # Expected price impact (simplified)
        # In production: use actual volume data
        base_impact = 0.001  # 0.1% base impact

        while remaining_qty > 0:
            remaining_time = (params.end_time - datetime.now()).total_seconds()

            if remaining_time <= 0:
                break

            # Calculate urgency
            alpha = self.calculate_urgency(
                remaining_time,
                remaining_qty,
                params.quantity
            )

            # Get current price
            current_price = market_data.get('price', params.limit_price)

            if not current_price:
                break

            # Calculate optimal slice size
            urgency_factor = alpha * remaining_qty / params.quantity
            slice_size = int(remaining_qty * (0.1 + 0.4 * urgency_factor))
            slice_size = max(1, min(slice_size, remaining_qty))

            # Calculate price limit (limit price with IS adjustment)
            if params.limit_price:
                limit = params.limit_price * (1 - alpha * 0.01)  # Willing to pay more if urgent
            else:
                limit = current_price

            # Execute
            fill_price = min(limit, current_price * 1.01) if params.side == 'BUY' else max(limit, current_price * 0.99)

            total_value += slice_size * fill_price
            remaining_qty -= slice_size

            logger.debug(f"IS: Filled {slice_size}, remaining: {remaining_qty}, alpha: {alpha:.2f}")

            time.sleep(30)

        total_filled = params.quantity - remaining_qty
        avg_price = total_value / total_filled if total_filled > 0 else 0

        # Calculate slippage vs arrival price
        arrival_price = market_data.get('arrival_price', params.limit_price or avg_price)
        slippage = ((avg_price - arrival_price) / arrival_price) * 100 if arrival_price > 0 else 0

        return ExecutionResult(
            order_id=f"IS_{params.symbol}_{int(time.time())}",
            status="FILLED" if remaining_qty == 0 else "PARTIAL",
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_value=total_value,
            slippage=slippage,
            execution_time=30
        )


class IcebergOrder:
    """Iceberg order - hide true order size."""

    def __init__(self):
        self.display_percentage = 0.2  # Show 20% of order
        self.min_display = 100  # Minimum 100 shares

    def execute(self, params: OrderParams) -> ExecutionResult:
        """Execute as iceberg (hidden) order."""
        display_qty = max(
            self.min_display,
            int(params.quantity * self.display_percentage)
        )

        total_filled = 0
        total_value = 0.0
        iterations = 0
        max_iterations = 20

        logger.info(f"ICEBERG: Hidden order {params.quantity}, showing {display_qty}")

        while total_filled < params.quantity and iterations < max_iterations:
            current_price = self.client.get_ltp(params.symbol) or params.limit_price

            if not current_price:
                break

            # Reveal portion
            remaining = params.quantity - total_filled
            reveal = min(display_qty, remaining)

            # Execute revealed portion
            try:
                fill_price = current_price * (1 + 0.0002 if params.side == 'BUY' else -0.0002)
                total_filled += reveal
                total_value += reveal * fill_price

                logger.debug(f"ICEBERG: Revealed {reveal}, filled: {total_filled}")

            except Exception as e:
                logger.error(f"ICEBERG execution error: {e}")

            time.sleep(60)  # Wait between reveals
            iterations += 1

        avg_price = total_value / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            order_id=f"ICEBERG_{params.symbol}_{int(time.time())}",
            status="FILLED" if total_filled == params.quantity else "PARTIAL",
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_value=total_value,
            slippage=0,
            execution_time=iterations * 60
        )


class ExecutionOptimizer:
    """Select best execution algorithm based on conditions."""

    def __init__(self, client):
        self.client = client
        self.twap = TWAPExecutor(client)
        self.vwap = VWAPExecutor(client)
        self.is_algo = ImplementationShortfall()
        self.iceberg = IcebergOrder()

    def select_algorithm(self, order_size: int, volatility: str,
                        urgency: str, liquidity: str) -> str:
        """Select best execution algorithm."""
        # Large orders -> Iceberg or VWAP
        if order_size > 5000:
            return "VWAP"

        # High urgency -> IS
        if urgency == "HIGH":
            return "IS"

        # Low volatility -> TWAP
        if volatility == "LOW":
            return "TWAP"

        # Low liquidity -> Iceberg
        if liquidity == "LOW":
            return "ICEBERG"

        # Default
        return "VWAP"

    def execute_order(self, params: OrderParams, market_data: Dict = None) -> ExecutionResult:
        """Execute order with best algorithm."""
        if market_data is None:
            market_data = {'price': params.limit_price}

        algo = params.algo

        if algo == "TWAP":
            return self.twap.execute(params)
        elif algo == "VWAP":
            return self.vwap.execute(params)
        elif algo == "IS":
            return self.is_algo.execute(params, market_data)
        elif algo == "ICEBERG":
            return self.iceberg.execute(params)
        else:
            # Market order (immediate)
            return ExecutionResult(
                order_id=f"MARKET_{params.symbol}_{int(time.time())}",
                status="FILLED",
                filled_quantity=params.quantity,
                avg_fill_price=params.limit_price or 0,
                total_value=params.quantity * (params.limit_price or 0),
                slippage=0,
                execution_time=0
            )


class TransactionCostOptimizer:
    """Optimize transaction costs."""

    def __init__(self):
        self.broker_rates = {
            'zerodha': {'equity': 0.001, 'fno': 0.001},  # 0.1%
            'upstox': {'equity': 0.001, 'fno': 0.001},
            'angel': {'equity': 0.0015, 'fno': 0.001}
        }

    def calculate_costs(self, order_value: float, broker: str = 'zerodha',
                       exchange: str = 'NSE', product: str = 'CNC') -> Dict:
        """Calculate all transaction costs."""
        rates = self.broker_rates.get(broker, self.broker_rates['zerodha'])

        # Brokerage
        brokerage = min(20, order_value * rates.get('equity', 0.001))

        # STT (Securities Transaction Tax)
        stt = order_value * 0.001 if product == 'CNC' else 0.0005

        # GST
        gst = (brokerage + 0) * 0.18

        # SEBI charges
        sebi = order_value * 0.000015

        # Stamp duty (varies by state, assume 0.01%)
        stamp_duty = order_value * 0.0001

        total_cost = brokerage + stt + gst + sebi + stamp_duty

        return {
            'brokerage': brokerage,
            'stt': stt,
            'gst': gst,
            'sebi': sebi,
            'stamp_duty': stamp_duty,
            'total': total_cost,
            'cost_percent': (total_cost / order_value) * 100 if order_value > 0 else 0
        }

    def optimize_for_cost(self, order_value: float, max_cost_percent: float = 0.5) -> Dict:
        """Determine if order should be split to minimize costs."""
        costs = self.calculate_costs(order_value)

        if costs['cost_percent'] <= max_cost_percent:
            return {'should_split': False, 'reason': 'Cost within limit'}

        # Recommend splitting
        split_factor = math.ceil(costs['cost_percent'] / max_cost_percent)

        return {
            'should_split': True,
            'split_into': split_factor,
            'size_per_split': order_value / split_factor,
            'reason': f"Cost {costs['cost_percent']:.2f}% exceeds {max_cost_percent}%"
        }