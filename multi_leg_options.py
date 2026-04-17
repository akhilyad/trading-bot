"""
MULTI-LEG OPTIONS STRATEGIES MODULE
Advanced options: Iron Condors, Butterflies, Vertical Spreads, Strangles, Straddles
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from logger import logger


@dataclass
class OptionLeg:
    """Single option leg."""
    type: str  # CALL or PUT
    action: str  # BUY or SELL
    strike: float
    quantity: int
    premium: float
    expiry_days: int

@dataclass
class OptionStrategy:
    """Complete options strategy with multiple legs."""
    name: str
    legs: List[OptionLeg]
    net_debit_credit: float  # Positive = debit, Negative = credit
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_of_profit: float


class BlackScholes:
    """Black-Scholes option pricing."""

    def __init__(self, risk_free_rate: float = 0.07):
        self.r = risk_free_rate

    def calculate(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate option price."""
        if T <= 0:
            return 0

        d1 = (math.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == 'call':
            price = S * self._norm_cdf(d1) - K * math.exp(-self.r * T) * self._norm_cdf(d2)
        else:
            price = K * math.exp(-self.r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

        return price

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class MultiLegOptions:
    """Execute complex multi-leg options strategies."""

    def __init__(self):
        self.bs = BlackScholes()
        self.active_positions = {}

    def iron_condor(self, spot_price: float, volatility: float = 0.25,
                   days_to_expiry: int = 30) -> OptionStrategy:
        """Iron Condor: Sell OTM call + put, Buy further OTM call + put."""
        # Parameters
        call_short_strike = spot_price * 1.05  # 5% OTM
        put_short_strike = spot_price * 0.95   # 5% OTM
        call_long_strike = spot_price * 1.10  # 10% OTM
        put_long_strike = spot_price * 0.90   # 10% OTM

        T = days_to_expiry / 365

        # Calculate premiums
        call_short_prem = self.bs.calculate(spot_price, call_short_strike, T, volatility, 'call')
        put_short_prem = self.bs.calculate(spot_price, put_short_strike, T, volatility, 'put')
        call_long_prem = self.bs.calculate(spot_price, call_long_strike, T, volatility, 'call')
        put_long_prem = self.bs.calculate(spot_price, put_long_strike, T, volatility, 'put')

        # Net credit
        net_credit = (call_short_prem + put_short_prem) - (call_long_prem + put_long_prem)

        # Max profit = net credit
        max_profit = net_credit * 100  # Assuming lot size 100

        # Max loss = width of spread - credit
        max_loss = ((call_short_strike - call_long_strike) - net_credit) * 100

        breakeven = [put_short_strike - net_credit, call_short_strike + net_credit]

        # Probability of profit (simplified)
        pop = 0.65  # Based on probability price stays within strikes

        legs = [
            OptionLeg('CALL', 'SELL', call_short_strike, 1, call_short_prem, days_to_expiry),
            OptionLeg('CALL', 'BUY', call_long_strike, 1, call_long_prem, days_to_expiry),
            OptionLeg('PUT', 'SELL', put_short_strike, 1, put_short_prem, days_to_expiry),
            OptionLeg('PUT', 'BUY', put_long_strike, 1, put_long_prem, days_to_expiry)
        ]

        return OptionStrategy(
            name='IRON_CONDOR',
            legs=legs,
            net_debit_credit=-net_credit,  # Negative = credit
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            probability_of_profit=pop
        )

    def butterfly_spread(self, spot_price: float, volatility: float = 0.25,
                        days_to_expiry: int = 30, direction: str = 'CALL') -> OptionStrategy:
        """Butterfly Spread: Buy 1 ITM, Sell 2 ATM, Buy 1 OTM."""
        atm_strike = round(spot_price / 10) * 10
        itm_strike = atm_strike - 50
        otm_strike = atm_strike + 50

        T = days_to_expiry / 365

        if direction == 'CALL':
            itm_prem = self.bs.calculate(spot_price, itm_strike, T, volatility, 'call')
            atm_prem = self.bs.calculate(spot_price, atm_strike, T, volatility, 'call')
            otm_prem = self.bs.calculate(spot_price, otm_strike, T, volatility, call='call')

            # Cost (debit)
            net_debit = itm_prem + otm_prem - 2 * atm_prem
            max_profit = (atm_strike - itm_strike) - net_debit
            max_loss = net_debit

            breakeven = [itm_strike + net_debit, otm_strike - net_debit]

        else:
            itm_prem = self.bs.calculate(spot_price, itm_strike, T, volatility, 'put')
            atm_prem = self.bs.calculate(spot_price, atm_strike, T, volatility, 'put')
            otm_prem = self.bs.calculate(spot_price, otm_strike, T, volatility, 'put')

            net_debit = itm_prem + otm_prem - 2 * atm_prem
            max_profit = (otm_strike - atm_strike) - net_debit
            max_loss = net_debit

            breakeven = [itm_strike + net_debit, otm_strike - net_debit]

        legs = [
            OptionLeg(direction, 'BUY', itm_strike, 1, itm_prem if direction == 'CALL' else itm_prem, days_to_expiry),
            OptionLeg(direction, 'SELL', atm_strike, 2, atm_prem, days_to_expiry),
            OptionLeg(direction, 'BUY', otm_strike, 1, otm_prem, days_to_expiry)
        ]

        return OptionStrategy(
            name=f'BUTTERFLY_{direction}',
            legs=legs,
            net_debit_credit=net_debit,
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            breakeven=breakeven,
            probability_of_profit=0.60
        )

    def vertical_spread(self, spot_price: float, volatility: float = 0.25,
                       days_to_expiry: int = 30, direction: str = 'BULL_CALL') -> OptionStrategy:
        """Vertical Spread: Buy lower strike, sell higher strike."""
        if direction == 'BULL_CALL':
            long_strike = spot_price * 0.98
            short_strike = spot_price * 1.05
        elif direction == 'BEAR_CALL':
            long_strike = spot_price * 1.02
            short_strike = spot_price * 1.08
        elif direction == 'BULL_PUT':
            long_strike = spot_price * 0.95
            short_strike = spot_price * 0.90
        else:  # BEAR_PUT
            long_strike = spot_price * 1.05
            short_strike = spot_price * 1.10

        T = days_to_expiry / 365

        # Calculate based on spread type
        if 'CALL' in direction:
            long_prem = self.bs.calculate(spot_price, long_strike, T, volatility, 'call')
            short_prem = self.bs.calculate(spot_price, short_strike, T, volatility, 'call')
        else:
            long_prem = self.bs.calculate(spot_price, long_strike, T, volatility, 'put')
            short_prem = self.bs.calculate(spot_price, short_strike, T, volatility, 'put')

        net_debit = long_prem - short_prem if 'BULL' in direction else short_prem - long_prem
        width = abs(short_strike - long_strike)

        max_profit = (width - net_debit) * 100
        max_loss = net_debit * 100
        breakeven = [long_strike + net_debit] if 'CALL' in direction else [long_strike - net_debit]

        leg_type = 'CALL' if 'CALL' in direction else 'PUT'
        leg_action = 'BUY' if 'BULL' in direction else 'SELL'

        legs = [
            OptionLeg(leg_action, leg_type if 'CALL' in direction else 'PUT', long_strike, 1, long_prem, days_to_expiry),
            OptionLeg(leg_action.replace('BUY', 'SELL') if 'BUY' in leg_action else 'BUY', leg_type, short_strike, 1, short_prem, days_to_expiry)
        ]

        return OptionStrategy(
            name=direction,
            legs=legs,
            net_debit_credit=net_debit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            probability_of_profit=0.60
        )

    def straddle(self, spot_price: float, volatility: float = 0.25,
                days_to_expiry: int = 30, direction: str = 'LONG') -> OptionStrategy:
        """Straddle: Buy call + put at same strike."""
        strike = round(spot_price / 10) * 10
        T = days_to_expiry / 365

        call_prem = self.bs.calculate(spot_price, strike, T, volatility, 'call')
        put_prem = self.bs.calculate(spot_price, strike, T, volatility, 'put')

        if direction == 'LONG':
            net_debit = call_prem + put_prem
            max_loss = net_debit
            max_profit = float('inf')
            breakeven = [strike - net_debit, strike + net_debit]
            pop = 0.35  # Low probability for long straddle
        else:
            net_credit = call_prem + put_prem
            max_profit = net_credit
            max_loss = float('inf')
            breakeven = [strike - net_credit, strike + net_credit]
            pop = 0.65

        legs = [
            OptionLeg('CALL', 'BUY' if direction == 'LONG' else 'SELL', strike, 1, call_prem, days_to_expiry),
            OptionLeg('PUT', 'BUY' if direction == 'LONG' else 'SELL', strike, 1, put_prem, days_to_expiry)
        ]

        return OptionStrategy(
            name=f'STRADDLE_{direction}',
            legs=legs,
            net_debit_credit=net_debit if direction == 'LONG' else -net_credit,
            max_profit=max_profit * 100,
            max_loss=max_loss * 100 if max_loss != float('inf') else max_loss,
            breakeven=breakeven,
            probability_of_profit=pop
        )

    def strangle(self, spot_price: float, volatility: float = 0.25,
                days_to_expiry: int = 30, direction: str = 'LONG') -> OptionStrategy:
        """Strangle: Buy OTM call + OTM put."""
        call_strike = spot_price * 1.05
        put_strike = spot_price * 0.95
        T = days_to_expiry / 365

        call_prem = self.bs.calculate(spot_price, call_strike, T, volatility, 'call')
        put_prem = self.bs.calculate(spot_price, put_strike, T, volatility, 'put')

        if direction == 'LONG':
            net_debit = call_prem + put_prem
            max_loss = net_debit
            max_profit = float('inf')
            breakeven = [put_strike - net_debit, call_strike + net_debit]
            pop = 0.30
        else:
            net_credit = call_prem + put_prem
            max_profit = net_credit
            max_loss = float('inf')
            breakeven = [put_strike - net_credit, call_strike + net_credit]
            pop = 0.70

        legs = [
            OptionLeg('CALL', 'BUY' if direction == 'LONG' else 'SELL', call_strike, 1, call_prem, days_to_expiry),
            OptionLeg('PUT', 'BUY' if direction == 'LONG' else 'SELL', put_strike, 1, put_prem, days_to_expiry)
        ]

        return OptionStrategy(
            name=f'STRANGLE_{direction}',
            legs=legs,
            net_debit_credit=net_debit if direction == 'LONG' else -net_credit,
            max_profit=max_profit * 100,
            max_loss=max_loss * 100 if max_loss != float('inf') else max_loss,
            breakeven=breakeven,
            probability_of_profit=pop
        )


class OptionsSignalGenerator:
    """Generate options trading signals based on market conditions."""

    def __init__(self):
        self.options_engine = MultiLegOptions()

    def get_recommended_strategy(self, spot_price: float, volatility: float,
                               regime: str, days_to_expiry: int = 30) -> List[OptionStrategy]:
        """Get recommended options strategy based on regime."""
        strategies = []

        if 'SIDEWAYS' in regime:
            # Iron condor for range-bound
            ic = self.options_engine.iron_condor(spot_price, volatility, days_to_expiry)
            strategies.append(ic)

        elif 'BULL' in regime:
            # Bull call spread or long call
            bull_spread = self.options_engine.vertical_spread(spot_price, volatility, days_to_expiry, 'BULL_CALL')
            strategies.append(bull_spread)

        elif 'BEAR' in regime:
            # Bear put spread
            bear_spread = self.options_engine.vertical_spread(spot_price, volatility, days_to_expiry, 'BEAR_PUT')
            strategies.append(bear_spread)

        elif 'HIGHVOL' in regime:
            # Long straddle for volatility
            straddles = self.options_engine.straddle(spot_price, volatility, days_to_expiry, 'LONG')
            strategies.append(straddles)

        elif 'LOWVOL' in regime:
            # Short straddle for income
            straddle_short = self.options_engine.straddle(spot_price, volatility, days_to_expiry, 'SHORT')
            strategies.append(straddle_short)

        return strategies

    def calculate_hedge_ratio(self, portfolio_value: float, options_cost: float) -> float:
        """Calculate how much portfolio to hedge with options."""
        if portfolio_value == 0:
            return 0

        hedge_ratio = min(0.2, options_cost / portfolio_value)  # Max 20% hedge
        return hedge_ratio