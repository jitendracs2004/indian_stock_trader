"""
Verify a 5-trading-day prediction against known actual market data.

Method:
- Download/load local PostgreSQL OHLCV data.
- Create technical features.
- Choose the most recent date with 5 completed future sessions.
- Train only on observations strictly before that date.
- Predict the next 5-session return.
- Compare forecast versus actual result.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from database import PostgresDatabase
from feature_engineering import FeatureEngineer
from nifty500_symbols import NIFTY_500_SYMBOLS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Full Nifty 500 universe. To verify a smaller set, slice this list,
# e.g. SYMBOLS = NIFTY_500_SYMBOLS[:20]
SYMBOLS = NIFTY_500_SYMBOLS

HORIZON_DAYS = 5

# Buy only when the prediction is large enough to potentially exceed
# transaction costs and normal estimation noise.
BUY_THRESHOLD = 0.005       # +0.50%
SELL_THRESHOLD = -0.005     # -0.50%

# Round-trip approximation:
# 0.03% transaction cost each side x 2 + 0.05% slippage each side x 2.
ROUND_TRIP_COST = 0.0016    # 0.16%


def ensure_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Make technical features and return model input fields.

    The FeatureEngineer feature list is instance state, so deduplicate
    the list before training.
    """
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_all_features(df.copy())

    excluded = {
        "date",
        "symbol",
        "target_1d",
        "target_5d",
        "target_20d",
        "target_binary_1d",
        "target_binary_5d",
        "target_binary_20d",
        "returns",
    }

    feature_cols = [
        column
        for column in dict.fromkeys(feature_engineer.get_feature_columns())
        if column in features_df.columns and column not in excluded
    ]

    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, feature_cols


def get_signal(predicted_return: float) -> str:
    """Convert predicted return into a simple action."""
    if predicted_return >= BUY_THRESHOLD:
        return "BUY"
    if predicted_return <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def evaluate_one_symbol(symbol: str, raw_df: pd.DataFrame) -> dict | None:
    """
    Evaluate the most recent fully observable 5-day forecast for one symbol.

    The target was created as:
        close[t + 5] / close[t] - 1

    The prediction origin must have five later rows with actual prices.
    """
    df = raw_df[raw_df["symbol"] == symbol].copy()

    if df.empty:
        logger.warning("%s: no PostgreSQL rows found.", symbol)
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(
        subset=["date"],
        keep="last",
    ).reset_index(drop=True)

    features_df, feature_cols = ensure_feature_columns(df)

    if len(features_df) < 300:
        logger.warning(
            "%s: only %d usable rows after indicators. "
            "Use at least 3–5 years of OHLCV data.",
            symbol,
            len(features_df),
        )
        return None

    # We need an origin row where 5 future trading sessions already exist.
    # Last valid origin index = final row - 5 sessions.
    prediction_index = len(features_df) - HORIZON_DAYS - 1

    if prediction_index < 200:
        logger.warning("%s: insufficient data for a clean train/test split.", symbol)
        return None

    # Everything strictly before the prediction origin is the training set.
    train_df = features_df.iloc[:prediction_index].copy()

    # The exact row whose 5-day return we want to forecast.
    prediction_row = features_df.iloc[[prediction_index]].copy()

    # Actual market outcome known after five sessions.
    actual_row = features_df.iloc[prediction_index + HORIZON_DAYS].copy()

    train_df = train_df.dropna(subset=feature_cols + ["target_5d"])
    prediction_row = prediction_row.dropna(subset=feature_cols)

    if train_df.empty or prediction_row.empty:
        logger.warning("%s: insufficient valid model rows.", symbol)
        return None

    X_train = train_df[feature_cols]
    y_train = train_df["target_5d"]

    X_prediction = prediction_row[feature_cols]

    # Fit scaling only on historical training data, never future data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_prediction_scaled = scaler.transform(X_prediction)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    predicted_return = float(model.predict(X_prediction_scaled)[0])

    start_close = float(prediction_row["close"].iloc[0])
    end_close = float(actual_row["close"])

    actual_return = (end_close / start_close) - 1.0
    predicted_close = start_close * (1.0 + predicted_return)

    signal = get_signal(predicted_return)

    predicted_up = predicted_return > 0
    actual_up = actual_return > 0
    direction_correct = predicted_up == actual_up

    # Realized strategy result after estimated round-trip cost.
    if signal == "BUY":
        strategy_return = actual_return - ROUND_TRIP_COST
    elif signal == "SELL":
        # Cash-only delivery trading: a SELL means "do not hold / exit".
        # It is not short selling.
        strategy_return = 0.0
    else:
        strategy_return = 0.0

    return {
        "symbol": symbol,
        "prediction_date": prediction_row["date"].iloc[0],
        "actual_end_date": actual_row["date"],
        "start_close_inr": start_close,
        "predicted_close_inr": predicted_close,
        "actual_close_inr": end_close,
        "predicted_5d_return_pct": predicted_return * 100,
        "actual_5d_return_pct": actual_return * 100,
        "absolute_error_pct_points": abs(predicted_return - actual_return) * 100,
        "signal": signal,
        "direction_correct": direction_correct,
        "strategy_return_after_cost_pct": strategy_return * 100,
        "training_rows": len(train_df),
        "feature_count": len(feature_cols),
    }


def print_report(report_df: pd.DataFrame) -> None:
    """Print individual results and aggregate performance."""
    if report_df.empty:
        print("No verification report could be produced.")
        return

    display_cols = [
        "symbol",
        "prediction_date",
        "actual_end_date",
        "start_close_inr",
        "predicted_close_inr",
        "actual_close_inr",
        "predicted_5d_return_pct",
        "actual_5d_return_pct",
        "absolute_error_pct_points",
        "signal",
        "direction_correct",
        "strategy_return_after_cost_pct",
    ]

    formatted = report_df[display_cols].copy()

    for column in [
        "start_close_inr",
        "predicted_close_inr",
        "actual_close_inr",
        "predicted_5d_return_pct",
        "actual_5d_return_pct",
        "absolute_error_pct_points",
        "strategy_return_after_cost_pct",
    ]:
        formatted[column] = formatted[column].map(lambda value: round(value, 2))

    print("\n" + "=" * 118)
    print("MOST RECENT COMPLETED 5-TRADING-DAY OUT-OF-SAMPLE VERIFICATION")
    print("=" * 118)
    print(formatted.to_string(index=False))

    direction_accuracy = report_df["direction_correct"].mean() * 100
    mae = report_df["absolute_error_pct_points"].mean()
    buy_df = report_df[report_df["signal"] == "BUY"]

    print("\n" + "-" * 118)
    print("SUMMARY")
    print("-" * 118)
    print(f"Symbols evaluated:              {len(report_df)}")
    print(f"Directional accuracy:           {direction_accuracy:.2f}%")
    print(f"Mean absolute forecast error:   {mae:.2f} percentage points")
    print(f"BUY signals generated:          {len(buy_df)}")

    if not buy_df.empty:
        profitable_buys = (
            buy_df["strategy_return_after_cost_pct"] > 0
        ).mean() * 100

        average_buy_return = buy_df[
            "strategy_return_after_cost_pct"
        ].mean()

        print(f"Profitable BUY signals:         {profitable_buys:.2f}%")
        print(f"Average BUY return after cost:  {average_buy_return:.2f}%")

    print("-" * 118)


def main() -> None:
    print("=" * 118)
    print("INDIAN STOCK TRADER — LAST-WEEK PREDICTION VERIFICATION")
    print("=" * 118)

    db = PostgresDatabase()

    if not db.test_connection():
        raise ConnectionError(
            "Cannot connect to PostgreSQL. Verify POSTGRES_* values in .env."
        )

    raw_data = db.get_market_data(symbols=SYMBOLS)

    if raw_data.empty:
        raise RuntimeError(
            "No price data exists in market_ohlcv. "
            "Run your Yahoo Finance → PostgreSQL ingestion first."
        )

    results = []

    for symbol in SYMBOLS:
        print(f"\nEvaluating: {symbol}")
        result = evaluate_one_symbol(symbol, raw_data)

        if result is not None:
            results.append(result)

    report_df = pd.DataFrame(results)

    if not report_df.empty:
        report_df.to_csv(
            "last_week_verification_report.csv",
            index=False,
        )

        print_report(report_df)

        print(
            "\nSaved detailed report: "
            "last_week_verification_report.csv"
        )


if __name__ == "__main__":
    main()