"""
EMERGENCY PROTOCOLS & STRESS TESTING
Flash crash protection, emergency liquidation, crash simulation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
from logger import logger


# =============================================================================
# EMERGENCY PROTOCOLS
# =============================================================================

class EmergencyProtocols:
    """Flash crash protection and emergency procedures."""

    def __init__(self):
        self.panic_mode = False
        self.emergency_history = []
        self.circuit_breakers_triggered = 0

        # Circuit breaker levels
        self.drops_to_trigger = {
            3: 'LEVEL_1',   # 3% drop - alert
            5: 'LEVEL_2',   # 5% drop - reduce exposure
            7: 'LEVEL_3',   # 7% drop - close all long
            10: 'LEVEL_4'   # 10% drop - full liquidation
        }

    def check_market_crash(self, nifty_change: float, positions: Dict) -> Dict:
        """Check if market is crashing and take action."""
        action = None

        for drop_level, level_name in sorted(self.drops_to_trigger.items()):
            if nifty_change <= -drop_level:
                action = {
                    'level': level_name,
                    'drop': nifty_change,
                    'action': self._get_action_for_level(level_name),
                    'timestamp': datetime.now()
                }
                break

        if action:
            self.circuit_breakers_triggered += 1
            self.emergency_history.append(action)
            logger.warning(f"MARKET CRASH: {action['level']} - {action['action']}")

        return action

    def _get_action_for_level(self, level: str) -> str:
        """Get action for each crash level."""
        actions = {
            'LEVEL_1': 'ALERT_ONLY',
            'LEVEL_2': 'REDUCE_EXPOSURE_50%',
            'LEVEL_3': 'CLOSE_ALL_LONG',
            'LEVEL_4': 'FULL_LIQUIDATION'
        }
        return actions.get(level, 'MONITOR')

    def emergency_hedge(self, portfolio_value: float, positions: Dict,
                       nifty_price: float) -> Optional[Dict]:
        """Create emergency hedge."""
        # Calculate long exposure
        long_exposure = sum(p.quantity * p.entry_price
                          for p in positions.values()
                          if p.type == 'LONG')

        if long_exposure == 0:
            return None

        # Hedge ratio based on exposure
        hedge_ratio = min(1.0, long_exposure / portfolio_value)

        # Buy NIFTY puts (simplified)
        strike = nifty_price * 0.95  # 5% OTM
        premium_estimate = nifty_price * 0.02  # ~2% premium

        return {
            'hedge_type': 'EMERGENCY_PUT',
            'strike': strike,
            'premium': premium_estimate,
            'hedge_ratio': hedge_ratio,
            'cost': premium_estimate * hedge_ratio,
            'reason': 'Emergency protection due to market conditions'
        }

    def panic_button(self, positions: Dict) -> Dict:
        """Execute panic button - close all positions."""
        self.panic_mode = True

        positions_to_close = []
        for symbol, position in positions.items():
            positions_to_close.append({
                'symbol': symbol,
                'type': position.type,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'entry_time': position.entry_time
            })

        logger.critical("PANIC BUTTON ACTIVATED - CLOSING ALL POSITIONS")

        return {
            'action': 'CLOSE_ALL',
            'positions': positions_to_close,
            'timestamp': datetime.now()
        }

    def cooldown_period(self, cooloff_minutes: int = 60) -> datetime:
        """Set cooldown period after emergency."""
        cooldown_until = datetime.now() + timedelta(minutes=cooloff_minutes)
        logger.info(f"Cooldown active until {cooldown_until}")
        return cooldown_until

    def check_recovery(self, nifty_change: float) -> bool:
        """Check if market has recovered enough to resume."""
        # Can resume if NIFTY recovered 50% of drop
        return nifty_change > -1.5  # Not more than 1.5% down


# =============================================================================
# STRESS TESTING MODULE
# =============================================================================

class StressTester:
    """Stress test trading system under extreme scenarios."""

    def __init__(self):
        self.scenarios = {}

    def define_scenarios(self) -> Dict[str, Dict]:
        """Define stress test scenarios."""
        return {
            'flash_crash': {
                'nifty_change': -8.0,
                'volatility_spike': 3.0,
                'duration': 30,  # minutes
                'description': '8% drop in 30 minutes'
            },
            'slow_crash': {
                'nifty_change': -5.0,
                'volatility_spike': 2.0,
                'duration': 3600,  # 1 hour
                'description': '5% drop over 1 hour'
            },
            'gap_down_open': {
                'nifty_change': -4.0,
                'volatility_spike': 2.5,
                'duration': 15,  # minutes
                'description': '4% gap down at open'
            },
            'volatility_spike': {
                'nifty_change': -2.0,
                'volatility_spike': 4.0,
                'duration': 120,
                'description': 'Volatility quadruples with mild price move'
            },
            'liquidity_crisis': {
                'nifty_change': -3.0,
                'volatility_spike': 2.0,
                'slippage': 0.05,  # 5% slippage
                'description': '5% slippage due to no liquidity'
            }
        }

    def run_scenario(self, scenario: Dict, positions: Dict,
                     capital: float) -> Dict:
        """Run a single stress scenario."""
        initial_value = capital

        # Simulate position impact
        position_impacts = []

        for symbol, position in positions.items():
            # Price move based on scenario
            price_change = scenario['nifty_change']

            # Volatility impact on stop losses
            vol_mult = scenario.get('volatility_spike', 1.0)

            # Check if stop loss hit
            if position.type == 'LONG':
                price_after = position.entry_price * (1 + price_change / 100)

                # Would stop be triggered?
                stop_distance = abs(position.entry_price - position.stop_loss)
                if price_after <= position.stop_loss:
                    # Stop hit
                    pnl = (position.stop_loss - position.entry_price) * position.quantity
                else:
                    # Hold position
                    pnl = (price_after - position.entry_price) * position.quantity

            elif position.type == 'SHORT':
                price_after = position.entry_price * (1 - price_change / 100)

                if price_after >= position.stop_loss:
                    pnl = (position.entry_price - position.stop_loss) * position.quantity
                else:
                    pnl = (position.entry_price - price_after) * position.quantity

            # Slippage impact
            slippage = scenario.get('slippage', 0)
            if slippage > 0:
                pnl *= (1 - slippage)

            position_impacts.append({
                'symbol': symbol,
                'pnl': pnl,
                'exit_price': price_after
            })

        total_pnl = sum(p['pnl'] for p in position_impacts)
        final_value = capital + total_pnl

        return {
            'scenario': scenario['description'],
            'initial_capital': initial_value,
            'final_capital': final_value,
            'pnl': total_pnl,
            'drawdown_percent': ((initial_value - final_value) / initial_value) * 100,
            'positions_affected': len(position_impacts),
            'worst_position': min(position_impacts, key=lambda x: x['pnl']) if position_impacts else None
        }

    def monte_carlo_stress(self, positions: Dict, capital: float,
                          simulations: int = 1000) -> Dict:
        """Run Monte Carlo stress tests."""
        scenarios = self.define_scenarios()

        results = {}
        for name, scenario in scenarios.items():
            result = self.run_scenario(scenario, positions, capital)
            results[name] = result

        # Worst case analysis
        worst_case = min(results.values(), key=lambda x: x['drawdown_percent'])

        return {
            'scenarios': results,
            'worst_case': worst_case,
            'max_drawdown': worst_case['drawdown_percent'],
            'recovery_needed': abs(worst_case['drawdown_percent'])
        }

    def get_portfolio_resilience(self, positions: Dict, capital: float) -> Dict:
        """Calculate portfolio resilience score."""
        if not positions:
            return {'resilience': 100, 'risk_level': 'MINIMAL'}

        # Diversification
        num_positions = len(positions)
        diversification_score = min(100, num_positions * 10)

        # Position concentration
        total_exposure = sum(p.quantity * p.entry_price for p in positions.values())
        concentration = (total_exposure / capital) if capital > 0 else 1

        if concentration < 0.3:
            concentration_score = 100
        elif concentration < 0.5:
            concentration_score = 70
        elif concentration < 0.7:
            concentration_score = 50
        else:
            concentration_score = 30

        # Stop loss proximity
        stop_scores = []
        for pos in positions.values():
            stop_dist = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
            stop_scores.append(stop_dist)

        avg_stop = sum(stop_scores) / len(stop_scores) if stop_scores else 2
        stop_score = min(100, avg_stop * 10)

        # Overall resilience
        resilience = (diversification_score * 0.3 +
                     concentration_score * 0.4 +
                     stop_score * 0.3)

        if resilience > 80:
            risk_level = 'LOW'
        elif resilience > 60:
            risk_level = 'MODERATE'
        else:
            risk_level = 'HIGH'

        return {
            'resilience': resilience,
            'risk_level': risk_level,
            'diversification': diversification_score,
            'concentration': concentration_score,
            'stop_distance': avg_stop
        }


# =============================================================================
# ALTERNATIVE ASSETS MODULE
# =============================================================================

class AlternativeAssets:
    """Trading alternative instruments: Futures, Currency, Commodities."""

    # NSE F&O instruments
    FUTURES_INSTRUMENTS = {
        'NIFTY': 'NIFTY',
        'BANKNIFTY': 'BANKNIFTY',
        'FINNIFTY': 'NIFTY FIN SERVICE',
        'MIDCAPNIFTY': 'NIFTY MIDCAP SELECT'
    }

    CURRENCY_PAIRS = ['USDINR', 'EURINR', 'GBPINR', 'JPYINR']
    COMMODITIES = ['GOLD', 'SILVER', 'CRUDE', 'NATURALGAS']

    def __init__(self):
        self.future_contracts = {}
        self.currency_contracts = {}
        self.commodity_contracts = {}

    def get_futures_data(self, symbol: str, expiry: str = 'current') -> Optional[Dict]:
        """Get futures data for index/stock futures."""
        # In production: fetch from broker's instrument list
        return {
            'symbol': symbol,
            'expiry': expiry,
            'lot_size': self._get_lot_size(symbol),
            'margin_required': self._get_margin(symbol),
            'contract_value': 0  # Current futures price
        }

    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for futures contract."""
        lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'FINNIFTY': 50,
            'MIDCAPNIFTY': 50
        }
        return lot_sizes.get(symbol, 50)

    def _get_margin(self, symbol: str) -> float:
        """Get margin required for futures."""
        margins = {
            'NIFTY': 150000,
            'BANKNIFTY': 125000,
            'FINNIFTY': 100000,
            'MIDCAPNIFTY': 70000
        }
        return margins.get(symbol, 100000)

    def calculate_futures_position(self, capital: float, risk_percent: float,
                                 entry: float, stop: float,
                                 lot_size: int) -> Dict:
        """Calculate futures position size."""
        risk_amount = capital * risk_percent
        risk_per_lot = abs(entry - stop) * lot_size

        if risk_per_lot == 0:
            return {'lots': 0, 'quantity': 0}

        max_lots = int(risk_amount / risk_per_lot)
        max_lots = min(max_lots, int(capital / (entry * lot_size)))

        return {
            'lots': max_lots,
            'quantity': max_lots * lot_size,
            'capital_required': max_lots * entry * lot_size,
            'risk_amount': max_lots * risk_per_lot
        }

    def futures_momentum_signal(self, futures_data: Dict,
                               spot_data: Dict) -> Optional[Dict]:
        """Generate futures momentum signal."""
        # Compare futures to spot (contango/backwardation)
        futures_price = futures_data.get('price', 0)
        spot_price = spot_data.get('price', 0)

        if futures_price == 0 or spot_price == 0:
            return None

        premium = ((futures_price - spot_price) / spot_price) * 100

        # Positive premium = contango, negative = backwardation
        if premium > 1.0:
            # Contango - expect spot to rise to futures
            return {
                'action': 'BUY_SPOT',
                'reason': f'Contango: futures {premium:.2f}% above spot'
            }
        elif premium < -1.0:
            # Backwardation - expect spot to fall
            return {
                'action': 'SHORT_SPOT',
                'reason': f'Backwardation: futures {premium:.2f}% below spot'
            }

        return None

    def get_currency_data(self, pair: str) -> Optional[Dict]:
        """Get currency pair data."""
        # In production: fetch from RBI reference rate or broker
        return {
            'pair': pair,
            'rate': 0,
            'change_24h': 0,
            'volume': 0
        }

    def currency_signal(self, usd_inr: float, previous: float,
                       nifty_change: float) -> Optional[Dict]:
        """Generate currency-related signal."""
        inr_change = ((usd_inr - previous) / previous) * 100

        # Strong INR = typically negative for NIFTY
        if inr_change > 0.5 and nifty_change < -0.5:
            return {
                'signal': 'CURRENCY_HEADWIND',
                'action': 'REDUCE_EXPOSURE',
                'reason': 'INR strengthening, expect pressure on NIFTY'
            }
        elif inr_change < -0.5 and nifty_change > 0.5:
            return {
                'signal': 'CURRENCY_TAILWIND',
                'action': 'INCREASE_EXPOSURE',
                'reason': 'INR weakening, tailwind for NIFTY'
            }

        return None

    def get_commodity_data(self, commodity: str) -> Optional[Dict]:
        """Get commodity data."""
        return {
            'commodity': commodity,
            'price': 0,
            'change': 0,
            'volume': 0
        }

    def commodity_correlation_signal(self, gold_change: float,
                                    nifty_change: float) -> Optional[Dict]:
        """Generate signal based on gold-NIFTY correlation."""
        # Gold typically has negative correlation with equity
        if gold_change > 2.0 and nifty_change < -1.0:
            return {
                'action': 'HEDGE_WITH_GOLD',
                'reason': 'Gold up, equity down - flight to safety'
            }
        elif gold_change < -2.0 and nifty_change > 1.0:
            return {
                'action': 'REDUCE_GOLD',
                'reason': 'Gold down, equity up - risk on'
            }

        return None


# =============================================================================
# EXECUTION OPTIMIZATION
# =============================================================================

class ExecutionOptimizer:
    """Optimize execution timing and order types."""

    def __init__(self):
        self.best_times = {
            'entry': [],  # Best entry times
            'exit': []    # Best exit times
        }

    def get_best_execution_time(self, action: str, regime: str) -> str:
        """Determine best time to execute."""
        # Avoid first 15 minutes (high volatility)
        # Avoid last 10 minutes (closing noise)

        if regime == 'BULL':
            return '10:00-14:00'  # Mid-morning to afternoon
        elif regime == 'BEAR':
            return '10:00-12:00'  # Morning only
        elif regime == 'SIDEWAYS':
            return '11:00-14:00'  # Mid-day range

        return '10:00-14:00'

    def should_use_limit_order(self, volatility: str, liquidity: str) -> bool:
        """Determine if limit order should be used."""
        if volatility == 'HIGH':
            return False  # Use market for speed
        elif liquidity == 'LOW':
            return True  # Use limit for better price
        else:
            return True  # Default to limit

    def split_order_size(self, total_quantity: int) -> List[int]:
        """Split large orders into smaller chunks."""
        if total_quantity <= 100:
            return [total_quantity]

        # Split into 5 parts
        part_size = total_quantity // 5
        remainder = total_quantity % 5

        sizes = [part_size] * 5
        for i in range(remainder):
            sizes[i] += 1

        return sizes

    def calculate_slippage_estimate(self, order_size: int,
                                    avg_volume: float) -> float:
        """Estimate slippage based on order size vs volume."""
        if avg_volume == 0:
            return 0.01  # 1% default

        volume_ratio = order_size / avg_volume

        if volume_ratio < 0.01:
            return 0.001  # 0.1%
        elif volume_ratio < 0.05:
            return 0.003  # 0.3%
        elif volume_ratio < 0.1:
            return 0.005  # 0.5%
        else:
            return 0.01  # 1%

    def bracket_order_params(self, entry: float, stop: float,
                           target: float, quantity: int) -> Dict:
        """Generate bracket order parameters."""
        return {
            'entry': entry,
            'stop_loss': stop,
            'target': target,
            'quantity': quantity,
            'order_type': 'BRACKET',
            'trailing_stop': (target - entry) * 0.3  # 30% of target
        }

    def amo_timing(self) -> Dict:
        """Get After Market Order timing."""
        return {
            'available': True,
            'order_window': '16:40-17:00',
            'validity': 'IOC',  # Immediate or Cancel
            'note': 'Orders executed at next day open'
        }


# =============================================================================
# PORTFOLIO OPTIMIZATION (Modern Portfolio Theory)
# =============================================================================

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""

    def __init__(self):
        self.returns_history = defaultdict(list)

    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate periodic returns."""
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        return returns

    def calculate_volatility(self, returns: List[float]) -> float:
        """Calculate portfolio volatility."""
        if not returns:
            return 0

        return np.std(returns) * np.sqrt(252)  # Annualized

    def calculate_correlation(self, returns1: List[float],
                            returns2: List[float]) -> float:
        """Calculate correlation between two return series."""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0

        return np.corrcoef(returns1, returns2)[0, 1]

    def build_correlation_matrix(self, price_data: Dict[str, List[float]]) -> pd.DataFrame:
        """Build correlation matrix from price data."""
        returns_data = {}

        for symbol, prices in price_data.items():
            returns_data[symbol] = self.calculate_returns(prices)

        # Create DataFrame
        returns_df = pd.DataFrame(returns_data)

        # Calculate correlation
        correlation = returns_df.corr()

        return correlation

    def optimize_weights(self, returns: Dict[str, List[float]],
                        target_return: float = None) -> Dict[str, float]:
        """Optimize portfolio weights for maximum Sharpe."""
        if not returns:
            return {}

        # Calculate mean returns
        mean_returns = {s: np.mean(r) * 252 for s, r in returns.items()}  # Annualized

        # For simplicity: equal weight with risk adjustment
        # Production: use scipy optimization
        num_assets = len(returns)
        equal_weight = 1.0 / num_assets

        # Adjust by inverse volatility
        volatilities = {s: self.calculate_volatility(r) * np.sqrt(252)
                      for s, r in returns.items()}

        inv_vol = {s: 1 / (v + 0.001) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        weights = {s: (inv_vol[s] / total_inv_vol) * equal_weight
                  for s in returns.keys()}

        return weights

    def risk_parity_allocation(self, volatilities: Dict[str, float],
                              total_capital: float) -> Dict[str, float]:
        """Risk parity allocation - equal risk contribution."""
        # Inverse volatility weighted
        inv_vol = {s: 1 / (v + 0.001) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        allocation = {}
        for symbol, inv in inv_vol.items():
            allocation[symbol] = (inv / total_inv_vol) * total_capital

        return allocation

    def max_sharpe_portfolio(self, returns: Dict[str, List[float]],
                            risk_free_rate: float = 0.07) -> Dict[str, float]:
        """Find portfolio with maximum Sharpe ratio."""
        # Simplified: use heuristic based on return/vol ratio
        sharpe_ratios = {}

        for symbol, ret in returns.items():
            if len(ret) > 1:
                mean_ret = np.mean(ret) * 252
                vol = self.calculate_volatility(ret) * np.sqrt(252)

                if vol > 0:
                    sharpe_ratios[symbol] = (mean_ret - risk_free_rate) / vol

        if not sharpe_ratios:
            return {s: 1/len(returns) for s in returns.keys()}

        # Weight by positive Sharpe ratios
        positive_sharpe = {s: max(0, r) for s, r in sharpe_ratios.items()}
        total = sum(positive_sharpe.values())

        if total == 0:
            return {s: 1/len(returns) for s in returns.keys()}

        weights = {s: (r / total) for s, r in positive_sharpe.items()}

        return weights