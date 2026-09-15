"""
Example Runner Script
Demonstrates how to use the trading system step-by-step
"""
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from settings import config
from data_ingestion import IndianMarketData
from feature_engineering import FeatureEngineer
from ml_models import StockPredictor
from backtester import Backtester
from risk_management import RiskManager
from database import PostgresDatabase

def main():
    """Run complete trading pipeline with detailed output"""

    print("="*70)
    print("INDIAN STOCK TRADING SYSTEM - EXAMPLE RUN")
    print("="*70)

    # Step 1: Initialize
    print("\n[STEP 1] Initializing components...")
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    data_handler = IndianMarketData(data_dir="data")
    feature_engineer = FeatureEngineer()
    predictor = StockPredictor(models_dir="models")
    backtester = Backtester(initial_capital=1000000)
    risk_manager = RiskManager(initial_capital=1000000)
    print(f"✓ Initialized with {len(symbols)} symbols")

    # Step 2: Fetch data
    print("\n[STEP 2] Fetching market data...")
    raw_data = data_handler.fetch_store_and_load(
        symbols=symbols,
        period="5y",
        refresh_from_yahoo=True,
    )

    if raw_data.empty:
        print("✗ No data available from PostgreSQL.")
        return

    print(f"✓ Loaded {len(raw_data)} total rows from PostgreSQL")
    print("  Table: market_ohlcv")

    # Step 3: Create features
    print("\n[STEP 3] Creating technical features...")
    processed_data = {}

    for symbol in symbols:
        symbol_df = raw_data[raw_data['symbol'] == symbol].copy()
        if symbol_df.empty:
            continue

        symbol_df = symbol_df.sort_values('date')
        symbol_df = feature_engineer.create_all_features(symbol_df)

        if data_handler.validate_data(symbol_df):
            processed_data[symbol] = symbol_df
            print(f"  ✓ {symbol}: {len(symbol_df)} rows, {len(feature_engineer.get_feature_columns())} features")

    # Step 4: Train models
    print("\n[STEP 4] Training ML models...")
    feature_cols = feature_engineer.get_feature_columns()
    all_data = pd.concat(processed_data.values(), ignore_index=True)

    metrics = predictor.train(
        df=all_data,
        feature_cols=feature_cols,
        target_col='target_5d',
        lookback=20,
        test_size=0.2
    )

    predictor.save_models(prefix="ensemble_predictor")

    print("\n  Training Results:")
    print(f"    LSTM RMSE: {metrics['lstm_rmse']:.6f}")
    print(f"    XGBoost RMSE: {metrics['xgb_rmse']:.6f}")
    print(f"    Ensemble RMSE: {metrics['ensemble_rmse']:.6f}")
    print(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"    Weights: LSTM={metrics['weights']['lstm']:.3f}, XGBoost={metrics['weights']['xgb']:.3f}")

    # Step 5: Generate signals (simplified example)
    print("\n[STEP 5] Generating trading signals...")

    # For demonstration, use simple moving average crossover
    for symbol, df in processed_data.items():
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['signal'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)

    print(f"  ✓ Generated signals for {len(processed_data)} symbols")

    db = PostgresDatabase()
    db.create_schema()

    signals_to_store = pd.DataFrame({
        "date": df["date"],
        "symbol": symbol,
        "prediction": 0.0,  # Replace with actual ML prediction later
        "signal": df["signal"],
    })

    saved_count = db.save_signals(
        signals_to_store,
        horizon="5d",
        model_name="sma_crossover_smoke_test",
        model_version="v1",
    )

    print(f"  ✓ Saved {saved_count} signals to PostgreSQL")



    # Step 6: Run backtest for one symbol (example)
    print("\n[STEP 6] Running backtest (example: RELIANCE.NS)...")

    if "RELIANCE.NS" in processed_data:
        test_df = processed_data["RELIANCE.NS"].copy()
        test_df = test_df.dropna(subset=['close', 'signal'])

        results = backtester.run_backtest(test_df, test_df['signal'], symbol="RELIANCE.NS")
        metrics = backtester.calculate_metrics(results)

        print("\n  Backtest Results for RELIANCE.NS:")
        print(f"    Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"    Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
        print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"    Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
        print(f"    Total Trades: {metrics.get('total_trades', 0)}")
        print(f"    Final Value: ₹{metrics.get('final_value_inr', 0):,.2f}")

        # Save results
        results.to_csv("data/backtest_results.csv", index=False)
        print(f"\n  ✓ Results saved to: data/backtest_results.csv")

    # Step 7: Risk metrics
    print("\n[STEP 7] Calculating risk metrics...")

    # Generate sample returns
    np.random.seed(42)
    sample_returns = pd.Series(np.random.randn(252) * 0.02 + 0.0005)
    risk_metrics = risk_manager.get_risk_metrics(sample_returns)

    print("\n  Risk Metrics (sample):")
    for key, value in risk_metrics.items():
        print(f"    {key}: {value}")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  - data/raw_market_data.csv (historical prices)")
    print("  - data/backtest_results.csv (backtest output)")
    print("  - models/ensemble_predictor_*.h5/.pkl (trained models)")
    print("\nNext steps:")
    print("  1. Review backtest results")
    print("  2. Adjust parameters in config/settings.py")
    print("  3. Run full pipeline: python trading_engine.py")
    print("  4. Deploy API: python -m uvicorn api.main:app --reload")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  - Check internet connection")
        print("  - Verify TA-Lib installation: pip show ta-lib")
        print("  - Check logs for detailed error messages")
        import traceback
        traceback.print_exc()
