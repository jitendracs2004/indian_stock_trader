"""
Machine Learning Models for Stock Prediction
LSTM for weekly, XGBoost for monthly, with ensemble voting
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import os
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    """
    Ensemble model combining LSTM and XGBoost for stock prediction
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.lstm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.lstm_scaler = MinMaxScaler()

        self.feature_cols = None
        self.target_col = None

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str, lookback: int = 20) -> Tuple:
        """
        Prepare data for LSTM and XGBoost models

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            lookback: Number of days for LSTM lookback

        Returns:
            X_lstm, X_xgb, y, scaler, lstm_scaler
        """
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Drop rows with NaN
        df = df.dropna(subset=feature_cols + [target_col])

        # Scale features
        X = df[feature_cols].values
        y = df[target_col].values

        X_scaled = self.scaler.fit_transform(X)

        # Create LSTM sequences
        X_lstm, y_lstm = [], []
        for i in range(lookback, len(X_scaled)):
            X_lstm.append(X_scaled[i-lookback:i])
            y_lstm.append(y[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # XGBoost uses current features (no lookback)
        X_xgb = X_scaled[lookback:]
        y_xgb = y[lookback:]

        logger.info(f"Created LSTM sequences: {X_lstm.shape}")
        logger.info(f"Created XGBoost data: {X_xgb.shape}")

        return X_lstm, X_xgb, y_xgb

    def build_lstm_model(self, input_shape: Tuple, units: int = 64) -> Sequential:
        """Build LSTM model for time-series prediction"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model

    def build_xgb_model(self, objective: str = 'reg:squarederror') -> xgb.XGBRegressor:
        """Build XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )

        logger.info("Built XGBoost model")
        return model

    def train(self, df: pd.DataFrame, feature_cols: List[str], 
              target_col: str = 'target_5d', lookback: int = 20,
              test_size: float = 0.2) -> dict:
        """
        Train both LSTM and XGBoost models

        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X_lstm, X_xgb, y = self.prepare_data(df, feature_cols, target_col, lookback)

        # Train-test split
        X_lstm_train, X_lstm_test, X_xgb_train, X_xgb_test, y_train, y_test =             train_test_split(X_lstm, X_xgb, y, test_size=test_size, shuffle=False)

        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

        # Train LSTM
        logger.info("Training LSTM model...")
        self.lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))

        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        history = self.lstm_model.fit(
            X_lstm_train, y_train,
            validation_data=(X_lstm_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        # LSTM predictions
        y_pred_lstm = self.lstm_model.predict(X_lstm_test).flatten()
        lstm_mse = mean_squared_error(y_test, y_pred_lstm)
        lstm_rmse = np.sqrt(lstm_mse)

        logger.info(f"LSTM Test RMSE: {lstm_rmse:.6f}")

        # Train XGBoost
        logger.info("Training XGBoost model...")
        self.xgb_model = self.build_xgb_model()

        self.xgb_model.fit(
            X_xgb_train, y_train,
            eval_set=[(X_xgb_test, y_test)],
            verbose=False
        )

        # XGBoost predictions
        y_pred_xgb = self.xgb_model.predict(X_xgb_test)
        xgb_mse = mean_squared_error(y_test, y_pred_xgb)
        xgb_rmse = np.sqrt(xgb_mse)

        logger.info(f"XGBoost Test RMSE: {xgb_rmse:.6f}")

        # Ensemble (weighted average)
        # Weight inversely proportional to error
        lstm_weight = 1 / lstm_rmse
        xgb_weight = 1 / xgb_rmse
        total_weight = lstm_weight + xgb_weight

        w_lstm = lstm_weight / total_weight
        w_xgb = xgb_weight / total_weight

        logger.info(f"Ensemble weights - LSTM: {w_lstm:.3f}, XGBoost: {w_xgb:.3f}")

        y_pred_ensemble = w_lstm * y_pred_lstm + w_xgb * y_pred_xgb
        ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
        ensemble_rmse = np.sqrt(ensemble_mse)

        logger.info(f"Ensemble Test RMSE: {ensemble_rmse:.6f}")

        # Direction accuracy
        ensemble_direction = (y_pred_ensemble > 0).astype(int)
        y_direction = (y_test > 0).astype(int)
        direction_accuracy = accuracy_score(y_direction, ensemble_direction)

        logger.info(f"Direction Accuracy: {direction_accuracy:.2%}")

        metrics = {
            'lstm_rmse': lstm_rmse,
            'xgb_rmse': xgb_rmse,
            'ensemble_rmse': ensemble_rmse,
            'direction_accuracy': direction_accuracy,
            'weights': {'lstm': w_lstm, 'xgb': w_xgb}
        }

        return metrics

    def predict(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Make prediction for the latest data point

        Args:
            df: DataFrame with features (must have at least lookback rows)
            lookback: Number of days for LSTM lookback

        Returns:
            Predicted return
        """
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Call train() first.")

        # Get last lookback rows
        X = df[self.feature_cols].values[-lookback:]
        X_scaled = self.scaler.transform(X)

        # LSTM prediction
        X_lstm = X_scaled.reshape(1, lookback, -1)
        pred_lstm = self.lstm_model.predict(X_lstm, verbose=0)[0, 0]

        # XGBoost prediction
        X_xgb = X_scaled[-1:].reshape(1, -1)
        pred_xgb = self.xgb_model.predict(X_xgb)[0]

        # Ensemble
        prediction = 0.5 * pred_lstm + 0.5 * pred_xgb

        return prediction

    def save_models(self, prefix: str = "stock_predictor"):
        """Save trained models to disk"""
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(self.models_dir, f"{prefix}_lstm.h5"))

        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, os.path.join(self.models_dir, f"{prefix}_xgb.pkl"))

        joblib.dump(self.scaler, os.path.join(self.models_dir, f"{prefix}_scaler.pkl"))
        joblib.dump(self.feature_cols, os.path.join(self.models_dir, f"{prefix}_features.pkl"))

        logger.info(f"Models saved to {self.models_dir}")

    def load_models(self, prefix: str = "stock_predictor"):
        """Load trained models from disk"""
        from tensorflow.keras.models import load_model

        lstm_path = os.path.join(self.models_dir, f"{prefix}_lstm.h5")
        xgb_path = os.path.join(self.models_dir, f"{prefix}_xgb.pkl")
        scaler_path = os.path.join(self.models_dir, f"{prefix}_scaler.pkl")
        features_path = os.path.join(self.models_dir, f"{prefix}_features.pkl")

        if os.path.exists(lstm_path):
            self.lstm_model = load_model(lstm_path)

        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if os.path.exists(features_path):
            self.feature_cols = joblib.load(features_path)

        logger.info(f"Models loaded from {self.models_dir}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(n)),
    })

    # Add some features
    for i in range(10):
        df[f'feature_{i}'] = np.random.randn(n)

    # Add target
    df['target_5d'] = np.roll(df['close'], -5) / df['close'] - 1

    # Train model
    predictor = StockPredictor(models_dir="models")
    feature_cols = [f'feature_{i}' for i in range(10)]

    metrics = predictor.train(df, feature_cols, 'target_5d', lookback=20, test_size=0.2)

    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save models
    predictor.save_models()
    print("\n✓ Models saved successfully")
