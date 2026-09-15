"""
Feature Engineering Module
Technical indicators, sentiment features, and fundamental ratios
Optimized for Indian market trading
"""
import pandas as pd
import numpy as np
import talib
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Create features for ML models from raw OHLCV data
    """

    def __init__(self):
        self.feature_columns = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in one call

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Technical indicators
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_pattern_features(df)

        # Target variables
        df = self.create_targets(df)

        # Drop NaN rows created by indicators
        df = df.dropna().reset_index(drop=True)

        logger.info(f"Created {len(self.feature_columns)} features")
        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Moving Averages
        df['sma_5'] = talib.SMA(close, timeperiod=5)
        df['sma_10'] = talib.SMA(close, timeperiod=10)
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)

        # EMA
        df['ema_5'] = talib.EMA(close, timeperiod=5)
        df['ema_10'] = talib.EMA(close, timeperiod=10)
        df['ema_20'] = talib.EMA(close, timeperiod=20)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

        # ADX (trend strength)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)

        # Aroon
        df['aroon_up'], df['aroon_down'] = talib.AROON(high, low, timeperiod=14)

        # Parabolic SAR
        df['psar'] = talib.SAR(high, low)

        self.feature_columns.extend([
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'aroon_up', 'aroon_down', 'psar'
        ])

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # RSI
        df['rsi_14'] = talib.RSI(close, timeperiod=14)
        df['rsi_7'] = talib.RSI(close, timeperiod=7)

        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)

        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

        # CCI
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)

        # ROC (Rate of Change)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=20)

        # Momentum
        df['mom_10'] = talib.MOM(close, timeperiod=10)

        self.feature_columns.extend([
            'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'willr',
            'cci', 'roc_10', 'roc_20', 'mom_10'
        ])

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Average True Range
        df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_7'] = talib.ATR(high, low, close, timeperiod=7)

        # Standard deviation
        df["std_20"] = df["close"].rolling(window=20, min_periods=20).std()

        # Historical volatility (annualized)
        df['returns'] = close / np.roll(close, 1) - 1
        df['hist_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        self.feature_columns.extend([
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
            'atr_14', 'atr_7', 'std_20', 'hist_vol_20'
        ])

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    
        """Add volume-based indicators."""

        # TA-Lib requires one-dimensional NumPy arrays with float64 dtype.
        close = np.ascontiguousarray(
            df["close"].astype("float64").to_numpy(),
            dtype=np.float64
        )
        high = np.ascontiguousarray(
            df["high"].astype("float64").to_numpy(),
            dtype=np.float64
        )
        low = np.ascontiguousarray(
            df["low"].astype("float64").to_numpy(),
            dtype=np.float64
        )
        volume = np.ascontiguousarray(
            df["volume"].astype("float64").to_numpy(),
            dtype=np.float64
        )

        # TA-Lib volume indicators
        df["obv"] = talib.OBV(close, volume)
        df["ad"] = talib.AD(high, low, close, volume)

        # Rolling volume statistics
        df["volume_sma_10"] = df["volume"].rolling(
            window=10,
            min_periods=10
        ).mean()

        df["volume_ratio"] = (
            df["volume"] / df["volume_sma_10"].replace(0, np.nan)
        )

        # Money Flow Index
        df["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)

        # Force Index:
        # (daily price change × volume), smoothed with a 13-day EMA
        raw_force_index = df["close"].diff() * df["volume"]
        df["force_index"] = raw_force_index.ewm(
            span=13,
            adjust=False
        ).mean()

        self.feature_columns.extend([
            "obv",
            "ad",
            "volume_sma_10",
            "volume_ratio",
            "mfi",
            "force_index",
        ])

        return df

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values

        # Price position relative to range
        df['price_range_pct'] = (close - low) / (high - low + 1e-10)

        # Gap (open vs previous close)
        df['gap'] = open_price / np.roll(close, 1) - 1

        # Candle patterns (simplified)
        df['body_size'] = abs(close - open_price) / (high - low + 1e-10)
        df['upper_shadow'] = (high - np.maximum(close, open_price)) / (high - low + 1e-10)
        df['lower_shadow'] = (np.minimum(close, open_price) - low) / (high - low + 1e-10)

        # Direction
        df['direction'] = np.where(close > open_price, 1, -1)

        # Higher highs / Lower lows
        df['higher_high'] = np.where(high > np.roll(high, 1), 1, 0)
        df['lower_low'] = np.where(low < np.roll(low, 1), 1, 0)

        self.feature_columns.extend([
            'price_range_pct', 'gap', 'body_size', 'upper_shadow',
            'lower_shadow', 'direction', 'higher_high', 'lower_low'
        ])

        return df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        close = df['close'].values

        # Next day return
        df['target_1d'] = np.roll(close, -1) / close - 1

        # 5-day return (weekly)
        df['target_5d'] = np.roll(close, -5) / close - 1

        # 20-day return (monthly)
        df['target_20d'] = np.roll(close, -20) / close - 1

        # Binary targets (up/down)
        df['target_binary_1d'] = (df['target_1d'] > 0).astype(int)
        df['target_binary_5d'] = (df['target_5d'] > 0).astype(int)
        df['target_binary_20d'] = (df['target_20d'] > 0).astype(int)

        # Remove last rows where targets are NaN
        df.loc[df.index[-20:], ['target_20d', 'target_binary_20d']] = np.nan
        df.loc[df.index[-5:], ['target_5d', 'target_binary_5d']] = np.nan
        df.loc[df.index[-1:], ['target_1d', 'target_binary_1d']] = np.nan

        self.feature_columns.extend([
            'target_1d', 'target_5d', 'target_20d',
            'target_binary_1d', 'target_binary_5d', 'target_binary_20d'
        ])

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns (excluding targets)"""
        target_cols = ['target_1d', 'target_5d', 'target_20d', 
                       'target_binary_1d', 'target_binary_5d', 'target_binary_20d']
        return [col for col in self.feature_columns if col not in target_cols]


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(n)),
        'high': 100 + np.cumsum(np.random.randn(n)) + np.abs(np.random.randn(n)),
        'low': 100 + np.cumsum(np.random.randn(n)) - np.abs(np.random.randn(n)),
        'close': 100 + np.cumsum(np.random.randn(n)),
        'volume': np.random.randint(100000, 1000000, n)
    })

    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_all_features(df)

    print(f"✓ Created {len(fe.get_feature_columns())} features")
    print(f"\nFeature columns: {fe.get_feature_columns()[:10]}...")
    print(f"\nSample data shape: {df_features.shape}")
    print(f"\nFirst few rows:")
    print(df_features[['close', 'rsi_14', 'macd', 'target_1d']].head())
