"""
ML MODELS MODULE
Machine Learning for signal prediction: XGBoost, Random Forest, LSTM, Meta-learner
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque
from logger import logger


class MLFeatureEngine:
    """Extract features for ML models."""

    def __init__(self):
        self.feature_cache = {}

    def extract_features(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Extract comprehensive features for ML model."""
        if df is None or len(df) < 50:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        features = {}

        # Price-based features
        for period in [1, 3, 5, 10, 20, 50]:
            if len(df) >= period:
                features[f'return_{period}d'] = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
                features[f'volatility_{period}d'] = close.pct_change().rolling(period).std().iloc[-1]

        # Moving average features
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                ma = close.rolling(period).mean().iloc[-1]
                features[f'price_to_ma_{period}'] = (close.iloc[-1] - ma) / ma

        # RSI
        if len(df) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi_14'] = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        if len(df) >= 26:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_hist'] = (macd - signal).iloc[-1]

        # Bollinger Bands
        if len(df) >= 20:
            bb = close.rolling(20)
            bb_mid = bb.mean()
            bb_std = bb.std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            features['bb_position'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 0.001)
            features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1]

        # ATR
        if len(df) >= 14:
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            features['atr_percent'] = (atr / close.iloc[-1]) * 100

        # Volume features
        if len(df) >= 20:
            features['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
            features['volume_trend'] = (volume.iloc[-1] - volume.iloc[-20]) / volume.iloc[-20]

        # Support/Resistance
        features['support_20'] = low.tail(20).min()
        features['resistance_20'] = high.tail(20).max()
        features['range_20'] = (features['resistance_20'] - features['support_20']) / close.iloc[-1]

        # Pattern features
        features['doji'] = 1 if abs(close.iloc[-1] - open.iloc[-1]) < (high.iloc[-1] - low.iloc[-1]) * 0.1 else 0
        features['hammer'] = 1 if (low.iloc[-1] < close.iloc[-1] * 0.98 and close.iloc[-1] > open.iloc[-1]) else 0

        # Trend strength
        if len(df) >= 50:
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50
            features['trend_strength'] = (ma50 - ma200) / ma200

        return features

    def create_dataset(self, historical_data: Dict[str, List], look_ahead: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training dataset from historical data."""
        X_data = []
        y_data = []

        for symbol, data in historical_data.items():
            df = pd.DataFrame(data)
            if len(df) < 50:
                continue

            close = df['close']

            for i in range(50, len(df) - look_ahead):
                features = self.extract_features(df.iloc[:i+1], symbol)

                if features:
                    # Future return as target
                    future_return = (close.iloc[i + look_ahead] - close.iloc[i]) / close.iloc[i]
                    label = 1 if future_return > 0.02 else (-1 if future_return < -0.02 else 0)

                    X_data.append(features)
                    y_data.append(label)

        X = pd.DataFrame(X_data)
        y = pd.Series(y_data)

        return X, y


class XGBoostModel:
    """XGBoost for signal prediction."""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model."""
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )

            # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
            y_mapped = y.map({-1: 0, 0: 1, 1: 2})

            self.model.fit(X, y_mapped)
            self.is_trained = True
            logger.info("XGBoost model trained")

        except ImportError:
            logger.warning("XGBoost not installed, using fallback")
            self.is_trained = False

    def predict(self, features: Dict) -> Tuple[str, float]:
        """Predict signal."""
        if not self.is_trained or not self.model:
            return "HOLD", 50.0

        try:
            X = pd.DataFrame([features])
            proba = self.model.predict_proba(X)[0]
            pred = self.model.predict(X)[0]

            # Map back: 0 -> SELL, 1 -> HOLD, 2 -> BUY
            labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            confidence = max(proba) * 100

            return labels[pred], confidence

        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return "HOLD", 50.0


class RandomForestModel:
    """Random Forest for signal prediction."""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            y_mapped = y.map({-1: 0, 0: 1, 1: 2})
            self.model.fit(X, y_mapped)
            self.is_trained = True

        except ImportError:
            logger.warning("sklearn not installed")

    def predict(self, features: Dict) -> Tuple[str, float]:
        """Predict signal."""
        if not self.is_trained or not self.model:
            return "HOLD", 50.0

        try:
            X = pd.DataFrame([features])
            proba = self.model.predict_proba(X)[0]
            pred = self.model.predict(X)[0]

            labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            confidence = max(proba) * 100

            return labels[pred], confidence

        except Exception as e:
            return "HOLD", 50.0


class LSTMModel:
    """LSTM for time-series prediction."""

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.model = None
        self.is_trained = False
        self.scaler = None

    def prepare_sequences(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM."""
        try:
            from sklearn.preprocessing import MinMaxScaler

            if self.scaler is None:
                self.scaler = MinMaxScaler()

            scaled_data = self.scaler.fit_transform(np.array(data).reshape(-1, 1))

            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:i + self.sequence_length])
                y.append(scaled_data[i + self.sequence_length])

            return np.array(X), np.array(y)

        except ImportError:
            return np.array([]), np.array([])

    def train(self, data: List[float]):
        """Train LSTM model."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            X, y = self.prepare_sequences(data)

            if len(X) == 0:
                return

            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])

            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            self.is_trained = True
            logger.info("LSTM model trained")

        except ImportError:
            logger.warning("TensorFlow not installed")

    def predict(self, recent_data: List[float]) -> Tuple[str, float]:
        """Predict next movement."""
        if not self.is_trained or not self.model or not recent_data:
            return "HOLD", 50.0

        try:
            from tensorflow.keras.preprocessing import sequence

            scaled = self.scaler.transform(np.array(recent_data[-self.sequence_length:]).reshape(-1, 1))
            X = sequence.pad_sequences([scaled], maxlen=self.sequence_length)
            X = X.reshape((1, self.sequence_length, 1))

            pred = self.model.predict(X, verbose=0)[0][0]

            # Threshold for signals
            if pred > 0.6:
                return "BUY", 70.0
            elif pred < 0.4:
                return "SELL", 70.0
            else:
                return "HOLD", 50.0

        except Exception as e:
            return "HOLD", 50.0


class MetaLearner:
    """Combine multiple ML models for better predictions."""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = deque(maxlen=50)

    def add_model(self, name: str, model):
        """Add a model to ensemble."""
        self.models[name] = model
        self.weights[name] = 1.0

    def predict(self, features: Dict) -> Tuple[str, float]:
        """Get weighted prediction from all models."""
        if not self.models:
            return "HOLD", 50.0

        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_confidence = 0

        for name, model in self.models.items():
            try:
                prediction, confidence = model.predict(features)

                # Adjust by model weight
                weight = self.weights.get(name, 1.0)
                votes[prediction] += confidence * weight
                total_confidence += confidence * weight

            except Exception as e:
                logger.error(f"Model {name} prediction error: {e}")

        if total_confidence == 0:
            return "HOLD", 50.0

        # Normalize
        for k in votes:
            votes[k] /= total_confidence

        # Return highest confidence
        best_signal = max(votes, key=votes.get)
        best_confidence = votes[best_signal] * 100

        return best_signal, best_confidence

    def update_weights(self, actual_result: str):
        """Update model weights based on performance."""
        self.performance_history.append(actual_result)

        if len(self.performance_history) < 10:
            return

        # Count correct predictions per model (simplified)
        for name in self.models:
            self.weights[name] *= 1.05 if actual_result == "WIN" else 0.95

        # Normalize weights
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total


class MLTradingEngine:
    """Complete ML trading system."""

    def __init__(self):
        self.feature_engine = MLFeatureEngine()
        self.xgboost = XGBoostModel()
        self.random_forest = RandomForestModel()
        self.lstm = LSTMModel()
        self.meta_learner = MetaLearner()

        self.is_trained = False

    def train_all(self, historical_data: Dict[str, List]):
        """Train all ML models."""
        logger.info("Training ML models...")

        X, y = self.feature_engine.create_dataset(historical_data)

        if len(X) < 100:
            logger.warning("Insufficient data for training")
            return

        # Train individual models
        try:
            self.xgboost.train(X, y)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

        try:
            self.random_forest.train(X, y)
        except Exception as e:
            logger.error(f"RF training failed: {e}")

        # Add to meta-learner
        self.meta_learner.add_model("XGBoost", self.xgboost)
        self.meta_learner.add_model("RandomForest", self.random_forest)

        self.is_trained = True
        logger.info("All ML models trained")

    def predict_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[str, int]:
        """Get prediction from meta-learner."""
        if not self.is_trained:
            # Use simpler model if not trained
            return "HOLD", 50

        features = self.feature_engine.extract_features(df, symbol)

        if not features:
            return "HOLD", 50

        signal, confidence = self.meta_learner.predict(features)

        return signal, int(confidence)

    def retrain_if_needed(self, new_data: Dict):
        """Periodically retrain with new data."""
        # Check if enough new data accumulated
        # In production: schedule periodic retraining
        pass