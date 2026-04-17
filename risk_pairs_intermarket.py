"""
RISK METRICS MODULE
Professional risk measures: VaR, CVaR, Factor Exposures, Beta
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime
from logger import logger


class ValueAtRisk:
    """Calculate Value at Risk (VaR) and Conditional VaR."""

    def __init__(self, confidence: float = 0.99, horizon: int = 1):
        self.confidence = confidence
        self.horizon = horizon
        self.returns_history = deque(maxlen=252)  # 1 year

    def update(self, returns: float):
        """Update returns history."""
        self.returns_history.append(returns)

    def calculate_var(self) -> float:
        """Calculate VaR using historical method."""
        if len(self.returns_history) < 30:
            return 0

        returns_array = np.array(self.returns_history)

        # Historical VaR
        var = np.percentile(returns_array, (1 - self.confidence) * 100)

        # Scale by horizon for longer periods
        var_scaled = var * np.sqrt(self.horizon)

        return abs(var_scaled) * 100  # Return as percentage

    def calculate_cvar(self) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(self.returns_history) < 30:
            return 0

        var = self.calculate_var()
        returns_array = np.array(self.returns_history)

        # CVaR = average of returns in worst (1-confidence)% cases
        tail_returns = returns_array[returns_array <= -var / 100]
        cvar = abs(tail_returns.mean()) * 100 if len(tail_returns) > 0 else var

        return cvar

    def calculate_parametric_var(self, mean_return: float = None, volatility: float = None) -> float:
        """Calculate parametric (Gaussian) VaR."""
        if mean_return is None or volatility is None:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            volatility = np.std(returns_array)

        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - self.confidence)

        # VaR = - (mean + z * vol) * sqrt(horizon)
        var = -(mean_return + z_score * volatility) * np.sqrt(self.horizon)

        return abs(var) * 100


class BetaCalculator:
    """Calculate portfolio beta to benchmark."""

    def __init__(self):
        self.returns_history = deque(maxlen=252)
        self.benchmark_returns = deque(maxlen=252)

    def update(self, portfolio_return: float, benchmark_return: float):
        """Update returns data."""
        self.returns_history.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)

    def calculate_beta(self) -> float:
        """Calculate beta."""
        if len(self.returns_history) < 30:
            return 1.0

        portfolio = np.array(self.returns_history)
        benchmark = np.array(self.benchmark_returns)

        # Beta = Cov(P, B) / Var(B)
        covariance = np.cov(portfolio, benchmark)[0][1]
        benchmark_variance = np.var(benchmark)

        if benchmark_variance == 0:
            return 1.0

        beta = covariance / benchmark_variance

        return beta

    def interpret_beta(self, beta: float) -> str:
        """Interpret beta value."""
        if beta > 1.5:
            return "HIGH_BETA"
        elif beta > 1.0:
            return "ABOVE_MARKET"
        elif beta > 0.8:
            return "BELOW_MARKET"
        elif beta > 0:
            return "LOW_BETA"
        else:
            return "INVERSE"


class FactorExposure:
    """Calculate factor exposures (Momentum, Value, Size, etc.)."""

    def __init__(self):
        self.factor_returns = {}

    def calculate_momentum(self, returns: List[float], lookback: int = 252) -> float:
        """Calculate momentum factor (past 12 months minus past month)."""
        if len(returns) < lookback:
            return 0

        recent = sum(returns[-lookback//12:])
        older = sum(returns[-lookback:-lookback//12])

        return recent - older

    def calculate_value(self, price_to_earnings: float, sector_avg: float) -> float:
        """Calculate value factor (PE vs sector)."""
        if sector_avg == 0:
            return 0
        return (sector_avg - price_to_earnings) / sector_avg

    def calculate_size(self, market_cap: float, index_market_cap: float) -> float:
        """Calculate size factor."""
        if index_market_cap == 0:
            return 0
        return np.log(market_cap / index_market_cap)

    def calculate_quality(self, roe: float, debt_equity: float) -> float:
        """Calculate quality factor (ROE - cost of debt)."""
        cost_of_debt = 0.08  # Assume 8%
        return roe - (cost_of_debt * debt_equity)

    def get_exposures(self, positions: Dict, market_data: Dict) -> Dict:
        """Calculate factor exposures for portfolio."""
        exposures = {
            'momentum': 0,
            'value': 0,
            'size': 0,
            'quality': 0
        }

        total_value = sum(p.entry_price * p.quantity for p in positions.values())

        if total_value == 0:
            return exposures

        for symbol, position in positions.items():
            position_value = position.entry_price * position.quantity
            weight = position_value / total_value

            # Get factor data for position
            md = market_data.get(symbol, {})

            exposures['momentum'] += weight * md.get('momentum', 0)
            exposures['value'] += weight * md.get('value', 0)
            exposures['size'] += weight * md.get('size', 0)
            exposures['quality'] += weight * md.get('quality', 0)

        return exposures


class RiskMetricsEngine:
    """Complete risk metrics engine."""

    def __init__(self):
        self.var_calculator = ValueAtRisk()
        self.beta_calculator = BetaCalculator()
        self.factor_exposure = FactorExposure()

    def calculate_all_metrics(self, portfolio_value: float, positions: Dict,
                            returns_history: List[float], benchmark_returns: List[float]) -> Dict:
        """Calculate comprehensive risk metrics."""
        # Update history
        for ret in returns_history[-5:]:
            self.var_calculator.update(ret)

        for ret, bench in zip(returns_history[-5:], benchmark_returns[-5:]):
            self.beta_calculator.update(ret, bench)

        # VaR and CVaR
        var_99 = self.var_calculator.calculate_var()
        cvar_99 = self.var_calculator.calculate_cvar()
        var_95 = self.var_calculator.calculate_var()

        # Beta
        beta = self.beta_calculator.calculate_beta()
        beta_interpretation = self.beta_calculator.interpret_beta(beta)

        # Position-level metrics
        max_position_risk = 0
        for symbol, position in positions.items():
            position_value = position.entry_price * position.quantity
            risk_percent = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
            max_position_risk = max(max_position_risk, risk_percent)

        return {
            'var_99': var_99,
            'var_95': var_95,
            'cvar_99': cvar_99,
            'beta': beta,
            'beta_interpretation': beta_interpretation,
            'max_position_risk': max_position_risk,
            'portfolio_concentration': max_position_risk,
            'risk_level': 'HIGH' if var_99 > 5 else 'MEDIUM' if var_99 > 2 else 'LOW'
        }


# =============================================================================
# PAIRS TRADING MODULE
# =============================================================================

class PairsTrading:
    """Statistical pairs trading."""

    def __init__(self):
        self.correlation_matrix = {}
        self.pairs = {}

    def calculate_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation between two return series."""
        if len(returns1) != len(returns2) or len(returns1) < 20:
            return 0

        return np.corrcoef(returns1, returns2)[0][1]

    def find_pairs(self, price_data: Dict[str, List[float]]) -> List[Tuple[str, str, float]]:
        """Find correlated pairs from price data."""
        returns = {}
        for symbol, prices in price_data.items():
            if len(prices) > 20:
                rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                returns[symbol] = rets

        pairs = []
        symbols = list(returns.keys())

        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = self.calculate_correlation(returns[symbols[i]], returns[symbols[j]])
                if corr > 0.7:  # Strong correlation
                    pairs.append((symbols[i], symbols[j], corr))

        # Sort by correlation
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs

    def calculate_spread(self, price1: float, price2: float, hedge_ratio: float) -> float:
        """Calculate spread between pair."""
        return price1 - hedge_ratio * price2

    def check_cointegration(self, price1: List[float], price2: List[float]) -> Dict:
        """Check if pair is cointegrated (mean reverting)."""
        if len(price1) != len(price2) or len(price1) < 30:
            return {'cointegrated': False, 'p_value': 1.0}

        # Simple Engle-Granger test (simplified)
        spread = np.array(price1) - np.array(price2)
        spread_returns = np.diff(spread)

        # Check if spread is mean-reverting
        half_life = -np.log(2) / np.mean(spread_returns / (spread[:-1] + 0.001))

        return {
            'cointegrated': half_life > 0 and half_life < 60,
            'half_life': half_life if half_life > 0 else 0,
            'spread_mean': np.mean(spread),
            'spread_std': np.std(spread)
        }

    def get_trading_signal(self, spread: float, historical_spread: List[float],
                          entry_threshold: float = 2.0) -> str:
        """Get trading signal from spread."""
        if len(historical_spread) < 20:
            return "HOLD"

        mean = np.mean(historical_spread)
        std = np.std(historical_spread)

        z_score = (spread - mean) / std if std > 0 else 0

        if z_score > entry_threshold:
            return "SHORT_SPREAD"  # Spread too high, expect to contract
        elif z_score < -entry_threshold:
            return "LONG_SPREAD"   # Spread too low, expect to expand

        return "HOLD"


class CalendarSpread:
    """Calendar spread trading (different expiry)."""

    def __init__(self):
        self.futures_data = {}

    def calculate_spread(self, near_expiry_price: float, far_expiry_price: float) -> float:
        """Calculate calendar spread."""
        return far_expiry_price - near_expiry_price

    def get_signal(self, spread: float, historical_spreads: List[float]) -> str:
        """Get signal from calendar spread."""
        if len(historical_spreads) < 10:
            return "HOLD"

        mean = np.mean(historical_spreads)
        std = np.std(historical_spreads)

        z_score = (spread - mean) / std if std > 0 else 0

        # Normal backwardation/contango adjustment
        if z_score > 1.5:
            return "SHORT_NEAR"  # Near month overvalued
        elif z_score < -1.5:
            return "LONG_NEAR"   # Near month undervalued

        return "HOLD"


# =============================================================================
# INTER-MARKET ANALYSIS MODULE
# =============================================================================

class InterMarketAnalysis:
    """Analyze cross-market relationships."""

    def __init__(self):
        self.correlations = {}

    def analyze_usd_inr(self, usd_inr: float, nifty: float,
                       prev_usd_inr: float, prev_nifty: float) -> Dict:
        """Analyze USD/INR vs NIFTY correlation."""
        usd_change = ((usd_inr - prev_usd_inr) / prev_usd_inr) * 100
        nifty_change = ((nifty - prev_nifty) / prev_nifty) * 100

        # Correlation analysis
        if usd_change > 0.5 and nifty_change < -0.5:
            signal = "STRONG_HEADWIND"
            action = "REDUCE_EXPOSURE"
            confidence = 75
        elif usd_change < -0.5 and nifty_change > 0.5:
            signal = "STRONG_TAILWIND"
            action = "INCREASE_EXPOSURE"
            confidence = 75
        elif usd_change > 0.2 and nifty_change < 0:
            signal = "HEADWIND"
            action = "CAUTIOUS"
            confidence = 60
        elif usd_change < -0.2 and nifty_change > 0:
            signal = "TAILWIND"
            action = "BULLISH"
            confidence = 60
        else:
            signal = "NEUTRAL"
            action = "NO_ACTION"
            confidence = 40

        return {
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'usd_change': usd_change,
            'nifty_change': nifty_change
        }

    def analyze_bonds_vs_financials(self, yield_change: float,
                                   financials_change: float) -> Dict:
        """Analyze bond yields vs financial sector."""
        # Higher yields = typically negative for financials short term
        if yield_change > 0.3 and financials_change < -1:
            return {
                'signal': 'BEARISH_FINANCIALS',
                'reason': 'Yield spike hurting financial stocks'
            }
        elif yield_change < -0.3 and financials_change > 1:
            return {
                'signal': 'BULLISH_FINANCIALS',
                'reason': 'Yield fall supporting financial stocks'
            }

        return {'signal': 'NEUTRAL'}

    def analyze_gold_vs_risk(self, gold_change: float, nifty_change: float) -> Dict:
        """Analyze gold vs risk assets."""
        # Negative correlation typically
        if gold_change > 2 and nifty_change < -1.5:
            return {
                'signal': 'FLIGHT_TO_SAFETY',
                'action': 'HEDGE_WITH_GOLD',
                'reason': 'Gold up, equities down - risk off'
            }
        elif gold_change < -2 and nifty_change > 1.5:
            return {
                'signal': 'RISK_ON',
                'action': 'REDUCE_GOLD',
                'reason': 'Gold down, equities up - risk on'
            }

        return {'signal': 'NEUTRAL'}

    def analyze_global_markets(self, nasdaq_change: float, nifty_change: float) -> Dict:
        """Analyze US markets impact on NIFTY."""
        # Global correlation
        if nasdaq_change > 1.5 and nifty_change > 0.5:
            return {
                'signal': 'GLOBAL_RALLY',
                'action': 'BULLISH',
                'reason': 'US markets supporting NIFTY'
            }
        elif nasdaq_change < -1.5 and nifty_change < -0.5:
            return {
                'signal': 'GLOBAL_SELL',
                'action': 'CAUTIOUS',
                'reason': 'US markets pressuring NIFTY'
            }
        elif nasdaq_change > 1 and nifty_change < 0:
            return {
                'signal': 'DIVERGENCE',
                'action': 'WAIT',
                'reason': 'US up, India down - mixed signals'
            }

        return {'signal': 'NEUTRAL'}


# =============================================================================
# FORWARD TESTING MODULE
# =============================================================================

class ForwardTesting:
    """Real-time strategy validation alongside live trading."""

    def __init__(self):
        self.paper_trades = []
        self.live_trades = []
        self.predictions = deque(maxlen=100)

    def record_prediction(self, symbol: str, predicted_action: str,
                         predicted_price: float, timeframe: str):
        """Record prediction for later comparison."""
        self.predictions.append({
            'symbol': symbol,
            'predicted_action': predicted_action,
            'predicted_price': predicted_price,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'actual_action': None,
            'actual_price': None
        })

    def record_actual(self, symbol: str, actual_action: str, actual_price: float):
        """Record actual outcome for predictions."""
        for pred in reversed(self.predictions):
            if pred['symbol'] == symbol and pred['actual_action'] is None:
                pred['actual_action'] = actual_action
                pred['actual_price'] = actual_price
                break

    def calculate_accuracy(self) -> Dict:
        """Calculate prediction accuracy."""
        completed = [p for p in self.predictions if p['actual_action'] is not None]

        if not completed:
            return {'accuracy': 0, 'total_predictions': 0}

        correct = sum(1 for p in completed if p['predicted_action'] == p['actual_action'])
        total = len(completed)

        # Price prediction accuracy
        price_errors = []
        for p in completed:
            if p['predicted_price'] and p['actual_price']:
                error = abs(p['predicted_price'] - p['actual_price']) / p['actual_price'] * 100
                price_errors.append(error)

        avg_price_error = sum(price_errors) / len(price_errors) if price_errors else 0

        return {
            'action_accuracy': correct / total * 100,
            'total_predictions': total,
            'avg_price_error_percent': avg_price_error,
            'predictions': completed[-10:]  # Last 10
        }

    def compare_strategies(self, strategy_a_results: List, strategy_b_results: List) -> Dict:
        """Compare two strategies A/B test."""
        if not strategy_a_results or not strategy_b_results:
            return {'winner': 'INSUFFICIENT_DATA'}

        a_returns = [r.get('return', 0) for r in strategy_a_results]
        b_returns = [r.get('return', 0) for r in strategy_b_results]

        a_avg = sum(a_returns) / len(a_returns)
        b_avg = sum(b_returns) / len(b_returns)

        a_win_rate = sum(1 for r in a_returns if r > 0) / len(a_returns)
        b_win_rate = sum(1 for r in b_returns if r > 0) / len(b_returns)

        return {
            'strategy_a_avg_return': a_avg,
            'strategy_b_avg_return': b_avg,
            'strategy_a_win_rate': a_win_rate * 100,
            'strategy_b_win_rate': b_win_rate * 100,
            'winner': 'A' if a_avg > b_avg else 'B'
        }

    def track_live_vs_paper(self, live_pnl: float, paper_pnl: float) -> Dict:
        """Track live vs paper performance."""
        return {
            'live_pnl': live_pnl,
            'paper_pnl': paper_pnl,
            'difference': live_pnl - paper_pnl,
            'divergence_percent': abs(live_pnl - paper_pnl) / abs(paper_pnl) * 100 if paper_pnl != 0 else 0
        }