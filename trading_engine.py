"""
Main Trading Engine
Orchestrates data ingestion, feature engineering, prediction, and execution
"""
import pandas as pd
import numpy as np
from datetime import datetime, time
import logging
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_ingestion import IndianMarketData
from features.feature_engineering import FeatureEngineer
from models.ml_models import StockPredictor
from backtest.backtester import Backtester
from utils.risk_management import RiskManager
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine for Indian stock market
    """

    def __init__(self, symbols: List[str] = None, initial_capital: float = 1000000):
        """
        Initialize trading engine

        Args:
            symbols: List of NSE symbols to trade
            initial_capital: Starting capital in INR
        """
        self.symbols = symbols or config.NIFTY_50_SYMBOLS[:10]  # Default: top 10 Nifty 50
        self.initial_capital = initial_capital

        # Initialize components
        self.data_handler = IndianMarketData(data_dir=config.DATA_DIR)
        self.feature_engineer = FeatureEngineer()
        self.predictor = StockPredictor(models_dir=config.MODELS_DIR)
        self.backtester = Backtester(initial_capital=initial_capital)
        self.risk_manager = RiskManager(initial_capital=initial_capital)

        # State
        self.is_trained = False
        self.current_positions = {}
        self.portfolio_value = initial_capital

        logger.info(f"Trading engine initialized with {len(self.symbols)} symbols")

    def fetch_and_prepare_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols and prepare features

        Returns:
            Dictionary of symbol -> DataFrame with features
        """
        logger.info("Fetching market data...")

        # Fetch data
        raw_data = self.data_handler.fetch_store_and_load(
            symbols=self.symbols,
            period=period,
            refresh_from_yahoo=True,
        )

        if raw_data.empty:
            logger.error("No data fetched. Check symbols and data source.")
            return {}

        # Save raw data
        logger.info("Raw OHLCV data is stored in PostgreSQL table: market_ohlcv")
        
        # Process each symbol
        processed_data = {}
        for symbol in self.symbols:
            symbol_df = raw_data[raw_data['symbol'] == symbol].copy()

            if symbol_df.empty:
                continue

            # Sort by date
            symbol_df = symbol_df.sort_values('date')

            # Create features
            symbol_df = self.feature_engineer.create_all_features(symbol_df)

            # Validate
            if self.data_handler.validate_data(symbol_df):
                processed_data[symbol] = symbol_df
                logger.info(f"✓ {symbol}: {len(symbol_df)} rows, {len(self.feature_engineer.get_feature_columns())} features")

        return processed_data

    def train_models(self, data: Dict[str, pd.DataFrame], target: str = 'target_5d'):
        """
        Train prediction models on all symbols

        Args:
            data: Dictionary of symbol -> DataFrame with features
            target: Target column to predict
        """
        logger.info("Training prediction models...")

        feature_cols = self.feature_engineer.get_feature_columns()

        # Combine all symbols for training (more data)
        all_data = pd.concat(data.values(), ignore_index=True)

        if all_data.empty:
            logger.error("No data available for training")
            return

        # Train ensemble model
        metrics = self.predictor.train(
            df=all_data,
            feature_cols=feature_cols,
            target_col=target,
            lookback=20,
            test_size=0.2
        )

        # Save models
        self.predictor.save_models(prefix="ensemble_predictor")

        self.is_trained = True

        logger.info("="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

        return metrics

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for all symbols

        Args:
            data: Dictionary of symbol -> DataFrame with features

        Returns:
            Dictionary of symbol -> signals DataFrame
        """

        from database import PostgresDatabase

        if not self.is_trained:
            logger.error("Models not trained. Call train_models() first.")
            return {}

        logger.info("Generating trading signals...")

        signals = {}
        feature_cols = self.feature_engineer.get_feature_columns()

        for symbol, df in data.items():
            # Make predictions
            predictions = []

            # Walk-forward prediction
            for i in range(20, len(df)):
                # Get lookback window
                lookback_df = df.iloc[i-20:i].copy()

                # Predict
                pred = self.predictor.predict(lookback_df, lookback=20)
                predictions.append(pred)

            # Create signals DataFrame
            signals_df = pd.DataFrame({
                'date': df['date'].iloc[20:].values,
                'symbol': symbol,
                'prediction': predictions,
                'signal': np.where(np.array(predictions) > 0.005, 1, 
                          np.where(np.array(predictions) < -0.005, -1, 0))
            })

            signals[symbol] = signals_df
            logger.info(f"✓ {symbol}: {len(signals_df)} signals generated")

        db = PostgresDatabase()
        db.create_schema()

        for symbol, signal_df in signals.items():
            db.save_signals(
                signal_df,
                horizon="5d",
                model_name="lstm_xgboost_ensemble",
                model_version="v1",
            )

        return signals

    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                     signals: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest on generated signals

        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Running backtest...")

        all_results = []
        all_metrics = {}

        for symbol in data.keys():
            if symbol not in signals:
                continue

            # Get price data and signals
            price_df = data[symbol].copy()
            signal_df = signals[symbol].copy()

            # Align dates
            signal_df = signal_df.set_index('date')
            price_df = price_df.set_index('date')

            # Merge signals with prices
            merged = price_df.join(signal_df[['signal']], how='inner')
            merged = merged.dropna(subset=['signal', 'close'])

            # Run backtest for this symbol
            self.backtester.reset()
            results = self.backtester.run_backtest(
                merged.reset_index(),
                merged['signal'],
                symbol=symbol
            )

            # Calculate metrics
            metrics = self.backtester.calculate_metrics(results)
            all_metrics[symbol] = metrics

            # Store results
            results['symbol'] = symbol
            all_results.append(results)

            logger.info(f"✓ {symbol}: {metrics.get('total_return_pct', 0):.2f}% return")

        # Combine results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            return {
                'results': combined_results,
                'metrics': all_metrics,
                'aggregate_metrics': self._aggregate_metrics(all_metrics)
            }

        return {}

    def _aggregate_metrics(self, metrics_dict: Dict) -> Dict:
        """Aggregate metrics across all symbols"""
        if not metrics_dict:
            return {}

        # Average key metrics
        keys_to_average = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']

        aggregated = {}
        for key in keys_to_average:
            values = [m[key] for m in metrics_dict.values() if key in m and m[key] is not None]
            if values:
                aggregated[f'avg_{key}'] = round(np.mean(values), 2)

        aggregated['total_symbols'] = len(metrics_dict)
        aggregated['total_trades'] = sum(m.get('total_trades', 0) for m in metrics_dict.values())

        return aggregated

    def run_full_pipeline(self, period: str = "2y", target: str = 'target_5d'):
        """
        Run complete trading pipeline: fetch -> train -> predict -> backtest

        Args:
            period: Data period to fetch
            target: Target column for prediction
        """
        logger.info("="*60)
        logger.info("STARTING FULL TRADING PIPELINE")
        logger.info("="*60)

        # Step 1: Fetch and prepare data
        data = self.fetch_and_prepare_data(period)

        if not data:
            logger.error("Pipeline stopped: No data available")
            return

        # Step 2: Train models
        self.train_models(data, target=target)

        # Step 3: Generate signals
        signals = self.generate_signals(data)

        # Step 4: Run backtest
        backtest_results = self.run_backtest(data, signals)

        # Step 5: Display results
        if backtest_results:
            self._display_results(backtest_results)

        return backtest_results

    def _display_results(self, results: Dict):
        """Display backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        metrics = results.get('metrics', {})
        aggregate = results.get('aggregate_metrics', {})

        print("\nPer-Symbol Performance:")
        print("-" * 60)
        for symbol, sym_metrics in metrics.items():
            print(f"\n{symbol}:")
            print(f"  Return: {sym_metrics.get('total_return_pct', 0):.2f}%")
            print(f"  Sharpe: {sym_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max DD: {sym_metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"  Trades: {sym_metrics.get('total_trades', 0)}")

        print("\n" + "-" * 60)
        print("Aggregate Performance:")
        print("-" * 60)
        for key, value in aggregate.items():
            print(f"  {key}: {value}")

        print("="*60)


# Example usage
if __name__ == "__main__":
    # Initialize engine with sample symbols
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

    engine = TradingEngine(symbols=symbols, initial_capital=1000000)

    # Run full pipeline
    results = engine.run_full_pipeline(period="1y", target='target_5d')

    if results:
        print("\n✓ Pipeline completed successfully!")
    else:
        print("\n✗ Pipeline failed. Check logs for details.")
