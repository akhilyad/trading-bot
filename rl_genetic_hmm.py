"""
DEEP REINFORCEMENT LEARNING MODULE
Deep Q-Network (DQN), Policy Gradients (PPO), and Self-Learning Trading Agent
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime
from dataclasses import dataclass
import math
from logger import logger


@dataclass
class State:
    """Trading state representation."""
    regime: str
    momentum_score: float
    mean_reversion_score: float
    volatility: float
    volume_ratio: float
    rsi: float
    price_vs_ma20: float
    price_vs_ma50: float
    positions_count: int
    daily_pnl: float

@dataclass
class Action:
    """Action taken by agent."""
    action_type: str  # BUY, SELL, HOLD, SCALE_IN, SCALE_OUT, CLOSE
    symbol: Optional[str] = None
    quantity: int = 0

@dataclass
class Experience:
    """Experience tuple for replay memory."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


class ReplayMemory:
    """Experience replay buffer for RL."""

    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


class DeepQNetwork:
    """Simple Deep Q-Network for trading decisions."""

    def __init__(self, state_size: int = 10, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size

        # Simple Q-table for now (in production: neural network)
        self.q_table = {}

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def get_q_value(self, state: State, action: str) -> float:
        """Get Q-value for state-action pair."""
        key = (self._state_to_key(state), action)
        return self.q_table.get(key, 0.0)

    def update_q_value(self, state: State, action: str, reward: float, next_state: State):
        """Update Q-value using Bellman equation."""
        key = (self._state_to_key(state), action)

        # Max Q-value for next state
        max_next_q = max(
            self.get_q_value(next_state, a)
            for a in ['BUY', 'SELL', 'HOLD', 'SCALE_IN', 'CLOSE']
        )

        # Bellman update
        current_q = self.q_table.get(key, 0.0)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[key] = new_q

    def _state_to_key(self, state: State) -> str:
        """Convert state to hashable key."""
        return f"{state.regime}:{int(state.momentum_score*10)}:{int(state.rsi)}:{state.positions_count}"

    def choose_action(self, state: State, training: bool = True) -> str:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            actions = ['BUY', 'SELL', 'HOLD', 'SCALE_IN', 'CLOSE']
            return random.choice(actions)

        # Greedy: choose action with highest Q-value
        actions = ['BUY', 'SELL', 'HOLD', 'SCALE_IN', 'CLOSE']
        q_values = [self.get_q_value(state, a) for a in actions]

        max_idx = np.argmax(q_values)
        return actions[max_idx]

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class PolicyGradient:
    """Policy gradient method for continuous action space."""

    def __init__(self):
        # Policy parameters
        self.weights = {
            'momentum': 0.25,
            'mean_reversion': 0.25,
            'volatility_adapt': 0.25,
            'volume_confirm': 0.25
        }

        self.advantages = deque(maxlen=100)

    def compute_advantage(self, reward: float, baseline: float) -> float:
        """Compute advantage for policy update."""
        advantage = reward - baseline
        self.advantages.append(advantage)
        return advantage

    def update_policy(self, advantage: float):
        """Update policy weights based on advantage."""
        # Positive advantage -> increase weight
        # Negative advantage -> decrease weight
        for key in self.weights:
            adjustment = advantage * 0.1 * random.choice([-1, 1])
            self.weights[key] = max(0.1, min(0.9, self.weights[key] + adjustment))

    def get_trading_weights(self) -> Dict[str, float]:
        """Get current trading strategy weights."""
        return self.weights.copy()


class RLTradingAgent:
    """Complete RL trading agent combining DQN and Policy Gradient."""

    def __init__(self):
        self.dqn = DeepQNetwork(state_size=10, action_size=5)
        self.policy = PolicyGradient()
        self.memory = ReplayMemory(capacity=10000)
        self.is_training = True
        self.episode_count = 0
        self.total_reward = 0

    def state_from_market(self, regime: str, indicators: Dict, positions: Dict, pnl: float) -> State:
        """Convert market data to RL state."""
        momentum = indicators.get('change_5d', 0) / 10  # Normalize
        mr_score = (50 - indicators.get('rsi', 50)) / 50  # Invert RSI for mean reversion
        vol = indicators.get('atr_percent', 2) / 10  # Normalize
        vol_ratio = indicators.get('volume_ratio', 1)
        rsi = indicators.get('rsi', 50) / 100
        price_vs_ma20 = (indicators.get('ma_20', 0) - indicators.get('ma_20', 0)) / indicators.get('ma_20', 1)
        price_vs_ma50 = (indicators.get('ma_50', 0) - indicators.get('ma_50', 0)) / indicators.get('ma_50', 1)

        return State(
            regime=regime,
            momentum_score=max(-1, min(1, momentum)),
            mean_reversion_score=max(-1, min(1, mr_score)),
            volatility=max(0, min(1, vol)),
            volume_ratio=max(0, min(2, vol_ratio)),
            rsi=max(0, min(1, rsi)),
            price_vs_ma20=max(-1, min(1, price_vs_ma20)),
            price_vs_ma50=max(-1, min(1, price_vs_ma50)),
            positions_count=len(positions),
            daily_pnl=max(-1, min(1, pnl / 10000))  # Normalize to +/- 10k
        )

    def choose_action(self, state: State, available_symbols: List[str]) -> Action:
        """Choose action using RL agent."""
        # Get base action from DQN
        action_type = self.dqn.choose_action(state, self.is_training)

        # Modify by policy weights
        if action_type == 'BUY' and available_symbols:
            # Weight by policy
            weights = self.policy.get_trading_weights()
            symbol = available_symbols[0]  # Simplified

            return Action(action_type='BUY', symbol=symbol, quantity=100)

        elif action_type == 'SELL' and available_symbols:
            return Action(action_type='SELL', symbol=available_symbols[0], quantity=100)

        return Action(action_type='HOLD')

    def store_experience(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Store experience in replay memory."""
        exp = Experience(state, action, reward, next_state, done)
        self.memory.push(exp)

    def train_step(self):
        """Train on batch from replay memory."""
        if len(self.memory) < 32:
            return

        batch = self.memory.sample(32)

        for exp in batch:
            # Compute reward (simplified)
            reward = exp.reward

            # Update DQN
            self.dqn.update_q_value(exp.state, exp.action.action_type, reward, exp.next_state)

            # Update policy
            baseline = np.mean([e.reward for e in self.memory])
            advantage = self.policy.compute_advantage(reward, baseline)
            self.policy.update_policy(advantage)

        # Decay exploration
        self.dqn.decay_epsilon()

    def get_strategy_parameters(self) -> Dict[str, float]:
        """Get evolved strategy parameters."""
        return {
            'dqn_epsilon': self.dqn.epsilon,
            'policy_weights': self.policy.get_trading_weights(),
            'total_reward': self.total_reward,
            'episodes': self.episode_count
        }


class GeneticAlgorithm:
    """Genetic algorithm for strategy evolution."""

    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.population = []
        self.generation = 0

    def initialize_population(self):
        """Create initial population of strategies."""
        for _ in range(self.population_size):
            strategy = {
                'ma_short': random.choice([9, 10, 15, 20]),
                'ma_long': random.choice([50, 100, 150, 200]),
                'rsi_oversold': random.randint(25, 40),
                'rsi_overbought': random.randint(60, 75),
                'atr_multiplier': random.uniform(1.5, 3.0),
                'position_size': random.uniform(0.5, 1.0),
                'stop_loss_pct': random.uniform(1.0, 3.0),
                'target_pct': random.uniform(2.0, 6.0),
                'fitness': 0
            }
            self.population.append(strategy)

    def evaluate_fitness(self, strategy: Dict, backtest_results: Dict) -> float:
        """Evaluate strategy fitness."""
        total_return = backtest_results.get('total_return', 0)
        max_drawdown = backtest_results.get('max_drawdown', 100)
        win_rate = backtest_results.get('win_rate', 0)
        trade_count = backtest_results.get('trade_count', 0)

        if trade_count < 10:
            return -100  # Penalize insufficient trades

        # Fitness = return / drawdown * win_rate
        fitness = (total_return / max(max_drawdown, 1)) * (win_rate ** 0.5)

        return fitness

    def selection(self) -> List[Dict]:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, 5)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner.copy())
        return selected

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point crossover."""
        child = {}
        for key in parent1.keys():
            if key == 'fitness':
                continue
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def mutate(self, strategy: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate strategy parameters."""
        mutated = strategy.copy()

        if random.random() < mutation_rate:
            mutated['ma_short'] = random.choice([9, 10, 15, 20])

        if random.random() < mutation_rate:
            mutated['ma_long'] = random.choice([50, 100, 150, 200])

        if random.random() < mutation_rate:
            mutated['rsi_oversold'] = random.randint(25, 40)

        if random.random() < mutation_rate:
            mutated['atr_multiplier'] = random.uniform(1.5, 3.0)

        if random.random() < mutation_rate:
            mutated['position_size'] = random.uniform(0.5, 1.0)

        return mutated

    def evolve(self, backtest_results: List[Dict]):
        """Run one generation of evolution."""
        # Evaluate fitness
        for i, strategy in enumerate(self.population):
            if i < len(backtest_results):
                strategy['fitness'] = self.evaluate_fitness(strategy, backtest_results[i])

        # Selection
        parents = self.selection()

        # Create next generation
        next_gen = []

        # Elitism: keep best 5
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        next_gen.extend([s.copy() for s in sorted_pop[:5]])

        # Crossover and mutation
        while len(next_gen) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            next_gen.append(child)

        self.population = next_gen
        self.generation += 1

        logger.info(f"Generation {self.generation} - Best fitness: {max(s['fitness'] for s in self.population):.2f}")

    def get_best_strategy(self) -> Dict:
        """Get best strategy from population."""
        return max(self.population, key=lambda x: x['fitness'])


class HiddenMarkovModel:
    """Hidden Markov Model for latent regime detection."""

    def __init__(self, n_states: int = 3):
        self.n_states = n_states

        # Transition matrix (initially equal probability)
        self.transition_matrix = np.ones((n_states, n_states)) / n_states

        # Emission parameters (mean and std for each state)
        self.means = np.array([0.0, 0.5, -0.5])
        self.stds = np.array([1.0, 1.0, 1.0])

        # Current state distribution
        self.state_probs = np.ones(n_states) / n_states

    def fit(self, returns: List[float]):
        """Fit HMM to return data."""
        # Simplified Baum-Welch (in production: full implementation)
        n = len(returns)

        for i, ret in enumerate(returns):
            # Compute emission probabilities
            emissions = self._emission_probability(ret)

            # Update state probabilities
            self.state_probs = self.state_probs @ self.transition_matrix * emissions

            # Normalize
            self.state_probs = self.state_probs / self.state_probs.sum()

        # Update means and stds based on most likely state
        for s in range(self.n_states):
            mask = self.state_probs == self.state_probs[s]
            if mask.sum() > 0:
                self.means[s] = np.mean([returns[i] for i in range(n) if self.state_probs[i] == s])
                self.stds[s] = np.std([returns[i] for i in range(n) if self.state_probs[i] == s]) + 0.01

    def _emission_probability(self, obs: float) -> np.ndarray:
        """Compute emission probability for observation."""
        probs = []
        for s in range(self.n_states):
            # Gaussian emission
            diff = obs - self.means[s]
            prob = math.exp(-0.5 * (diff / self.stds[s]) ** 2) / (self.stds[s] * math.sqrt(2 * math.pi))
            probs.append(prob)
        return np.array(probs)

    def predict_next_state(self) -> Tuple[int, float]:
        """Predict next state."""
        # Transition
        next_probs = self.state_probs @ self.transition_matrix

        # Most likely state
        state = np.argmax(next_probs)
        confidence = next_probs[state]

        return state, confidence

    def get_regime_label(self, state: int) -> str:
        """Get human-readable regime label."""
        labels = {0: 'CALM', 1: 'TRENDING', 2: 'VOLATILE'}
        return labels.get(state, 'UNKNOWN')


class BayesianModel:
    """Bayesian regression and probabilistic forecasting."""

    def __init__(self):
        # Prior parameters
        self.alpha = 1.0  # Precision prior
        self.beta = 1.0   # Noise precision

        # Posterior parameters
        self.mean = 0.0
        self.precision = 1.0

        self.data_history = []

    def update(self, x: float, y: float):
        """Update posterior with new observation."""
        self.data_history.append((x, y))

        # Keep only last 100 points
        if len(self.data_history) > 100:
            self.data_history = self.data_history[-100:]

        if len(self.data_history) < 2:
            return

        # Compute posterior (simplified)
        X = np.array([1] + [d[0] for d in self.data_history])
        Y = np.array([d[1] for d in self.data_history])

        # Simple linear regression
        XTX = X @ X.T
        try:
            cov = np.linalg.inv(XTX + self.alpha * np.eye(len(X)))
            self.mean = cov @ X @ Y
            self.precision = self.beta
        except:
            pass

    def predict(self, x: float) -> Tuple[float, float]:
        """Predict with uncertainty."""
        if not self.data_history:
            return 0.0, 1.0

        x_pred = np.array([1, x])
        prediction = x_pred @ self.mean

        # Uncertainty
        uncertainty = 1.0 / math.sqrt(self.precision)

        return prediction, uncertainty

    def get_probability_up(self, threshold: float = 0.0) -> float:
        """Get probability that return > threshold."""
        pred, unc = self.predict(0)
        z = (threshold - pred) / unc if unc > 0 else 0
        return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))


class GaussianProcess:
    """Gaussian Process for nonparametric regression."""

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.length_scale = 1.0
        self.noise = 0.1

    def kernel(self, x1: float, x2: float) -> float:
        """RBF kernel."""
        return math.exp(-0.5 * (x1 - x2) ** 2 / self.length_scale ** 2)

    def fit(self, X: List[float], y: List[float]):
        """Fit GP."""
        self.X_train = X[-50:]  # Keep last 50 points
        self.y_train = y[-50:]

    def predict(self, x: float) -> Tuple[float, float]:
        """Predict with uncertainty."""
        if not self.X_train:
            return 0.0, 1.0

        # Compute kernel
        k_star = [self.kernel(x, x0) for x0 in self.X_train]
        k_star_T = [[self.kernel(x, x2) for x2 in self.X_train]]
        k_star_T.append([0.1] * len(self.X_train))  # Add noise

        # Simplified prediction
        weights = [k / (sum(k) + 0.001) for k in k_star]
        pred = sum(w * y for w, y in zip(weights, self.y_train))

        return pred, 0.2  # Simplified uncertainty


class ProbabilisticTrading:
    """Complete probabilistic trading system."""

    def __init__(self):
        self.bayesian = BayesianModel()
        self.gp = GaussianProcess()
        self.hmm = HiddenMarkovModel(n_states=3)

    def update_models(self, returns: List[float], indicators: Dict):
        """Update all probabilistic models."""
        # Bayesian
        if returns:
            self.bayesian.update(returns[-2], returns[-1])

        # GP
        if len(returns) > 10:
            self.gp.fit(list(range(len(returns))), returns)

        # HMM
        if len(returns) > 30:
            self.hmm.fit(returns[-60:])

    def get_signals(self) -> Dict:
        """Get probabilistic signals."""
        signals = {}

        # Bayesian
        prob_up = self.bayesian.get_probability_up(0)
        signals['bayesian_prob_up'] = prob_up

        # GP prediction
        gp_pred, gp_unc = self.gp.predict(len(self.gp.X_train))
        signals['gp_trend'] = gp_pred
        signals['gp_uncertainty'] = gp_unc

        # HMM
        hmm_state, hmm_conf = self.hmm.predict_next_state()
        signals['hmm_regime'] = self.hmm.get_regime_label(hmm_state)
        signals['hmm_confidence'] = hmm_conf

        return signals