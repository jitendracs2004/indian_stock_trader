"""
Weekly and monthly NSE stock prediction.

Weekly horizon:
    5 future trading sessions.

Monthly horizon:
    20 future trading sessions.

Data source:
    Local PostgreSQL table: market_ohlcv

Output:
    - Console table
    - weekly_monthly_predictions.csv
    - PostgreSQL model_signals table
"""

import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

# Windows terminals default to cp1252, which cannot encode symbols like
# the rupee sign (\u20b9). Force UTF-8 so console printing never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.preprocessing import StandardScaler

from database import PostgresDatabase
from feature_engineering import FeatureEngineer
from nifty500_symbols import NIFTY_500_SYMBOLS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Full Nifty 500 universe. To run a smaller test set, slice this list,
# e.g. SYMBOLS = NIFTY_500_SYMBOLS[:20]
SYMBOLS = NIFTY_500_SYMBOLS

# A week and month in trading-session terms, not calendar days.
HORIZONS = {
    "daily_1d": {
        "target_column": "target_1d",
        "trading_days": 1,

        # Daily prediction is primarily used as a timing filter.
        "buy_threshold": 0.0025,       # +0.25%
        "sell_threshold": -0.0025,     # -0.25%
    },

    "weekly_5d": {
        "target_column": "target_5d",
        "trading_days": 5,
        "buy_threshold": 0.0050,       # +0.50%
        "sell_threshold": -0.0050,     # -0.50%
    },

    "monthly_20d": {
        "target_column": "target_20d",
        "trading_days": 20,
        "buy_threshold": 0.0120,       # +1.20%
        "sell_threshold": -0.0120,     # -1.20%
    },
}

MINIMUM_TRAINING_ROWS = 300


def normalize_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Create technical indicators and choose safe model-input columns.

    Targets and identifiers are excluded. Infinite values produced by
    ratios are converted to NaN and removed before fitting/prediction.
    """
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(raw_df.copy())

    excluded = {
        "date",
        "symbol",
        "returns",
        "target_1d",
        "target_5d",
        "target_20d",
        "target_binary_1d",
        "target_binary_5d",
        "target_binary_20d",
    }

    feature_cols = [
        col
        for col in dict.fromkeys(engineer.get_feature_columns())
        if col in features_df.columns and col not in excluded
    ]

    features_df = features_df.replace([np.inf, -np.inf], np.nan)

    return features_df, feature_cols


def get_signal(predicted_return: float, buy_threshold: float, sell_threshold: float) -> int:
    """
    Return:
      1 = BUY
      0 = HOLD / cash
     -1 = SELL / exit existing delivery holding

    This is not a short-selling instruction.
    """
    if predicted_return >= buy_threshold:
        return 1

    if predicted_return <= sell_threshold:
        return -1

    return 0


def signal_name(signal: int) -> str:
    """Human-readable signal label."""
    return {
        1: "BUY",
        0: "HOLD",
        -1: "SELL / EXIT",
    }.get(signal, "UNKNOWN")


def train_and_predict_horizon(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[float, int]:
    """
    Train only on rows whose realized future target is known.

    Latest row normally has unknown target because 5 or 20 future
    sessions have not yet occurred. That latest row is used only to predict.
    """
    training_df = features_df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    latest_df = features_df.dropna(
        subset=feature_cols
    ).sort_values("date").tail(1).copy()

    if len(training_df) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"Only {len(training_df)} valid training rows available for "
            f"{target_col}. Load at least 3–5 years of daily data."
        )

    if latest_df.empty:
        raise ValueError("No latest feature row is available for prediction.")

    X_train = training_df[feature_cols]
    y_train = training_df[target_col]

    X_latest = latest_df[feature_cols]

    # Scaling must fit only historical train data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_latest_scaled = scaler.transform(X_latest)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    predicted_return = float(model.predict(X_latest_scaled)[0])

    return predicted_return, len(training_df)


def predict_one_symbol(symbol: str, all_raw_data: pd.DataFrame) -> list[dict]:
    """Create both weekly and monthly forecasts for one stock."""
    raw_df = all_raw_data[
        all_raw_data["symbol"] == symbol
    ].copy()

    if raw_df.empty:
        logger.warning("%s: no database data available.", symbol)
        return []

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df = (
        raw_df
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    features_df, feature_cols = normalize_features(raw_df)

    if features_df.empty:
        logger.warning("%s: no usable feature rows.", symbol)
        return []

    latest = (
        features_df
        .dropna(subset=feature_cols)
        .sort_values("date")
        .tail(1)
    )

    if latest.empty:
        logger.warning("%s: latest row has incomplete features.", symbol)
        return []

    latest_date = pd.Timestamp(latest["date"].iloc[0])
    latest_close = float(latest["close"].iloc[0])

    predictions = []

    for horizon_name, horizon_config in HORIZONS.items():
        target_col = horizon_config["target_column"]

        try:
            predicted_return, training_rows = train_and_predict_horizon(
                features_df=features_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )

            predicted_price = latest_close * (1 + predicted_return)

            signal = get_signal(
                predicted_return=predicted_return,
                buy_threshold=horizon_config["buy_threshold"],
                sell_threshold=horizon_config["sell_threshold"],
            )

            predictions.append({
                "symbol": symbol,
                "prediction_date": latest_date,
                "horizon": horizon_name,
                "trading_days": horizon_config["trading_days"],
                "latest_close_inr": latest_close,
                "predicted_return_pct": predicted_return * 100,
                "predicted_price_inr": predicted_price,
                "signal_code": signal,
                "signal": signal_name(signal),
                "training_rows": training_rows,
                "feature_count": len(feature_cols),
            })

        except Exception as exc:
            logger.exception(
                "%s: %s model failed: %s",
                symbol,
                horizon_name,
                exc,
            )

    return predictions


def save_predictions_to_database(
    db: PostgresDatabase,
    prediction_df: pd.DataFrame,
) -> None:
    """
    Save weekly and monthly predictions in PostgreSQL.

    PostgreSQL model_signals.signal is SMALLINT:
        1  = BUY
        0  = HOLD / CASH
       -1  = SELL / EXIT

    The human-readable 'signal' label is deliberately not sent to
    database.py because that would conflict with the numeric SQL column.
    """
    for horizon in prediction_df["horizon"].dropna().unique():
        horizon_df = prediction_df[
            prediction_df["horizon"] == horizon
        ].copy()

        # Create a brand-new clean DataFrame. This avoids accidental
        # duplicate columns caused by DataFrame.rename().
        db_signals = pd.DataFrame({
            "date": pd.to_datetime(horizon_df["prediction_date"]),
            "symbol": horizon_df["symbol"].astype(str),
            "prediction": (
                pd.to_numeric(
                    horizon_df["predicted_return_pct"],
                    errors="coerce",
                ) / 100.0
            ),
            "signal": pd.to_numeric(
                horizon_df["signal_code"],
                errors="coerce",
            ).fillna(0).astype(int),
        })

        # Verify required fields and remove invalid records before insert.
        db_signals = db_signals.dropna(
            subset=["date", "symbol", "prediction", "signal"]
        )

        if db_signals.empty:
            logger.warning(
                "%s: no valid prediction records available to store.",
                horizon,
            )
            continue

        saved = db.save_signals(
            signals_df=db_signals,
            horizon=horizon,
            model_name="xgboost_direct_horizon",
            model_version="v1",
        )

        logger.info(
            "%s: stored %s prediction records in model_signals.",
            horizon,
            saved,
        )
        
def print_predictions(prediction_df: pd.DataFrame) -> None:
    """Print clean weekly and monthly tables."""
    if prediction_df.empty:
        print("No predictions were generated.")
        return

    for horizon in ["weekly_5d", "monthly_20d"]:
        horizon_df = prediction_df[
            prediction_df["horizon"] == horizon
        ].copy()

        if horizon_df.empty:
            continue

        horizon_df = horizon_df.sort_values(
            "predicted_return_pct",
            ascending=False,
        )

        display = horizon_df[
            [
                "symbol",
                "prediction_date",
                "latest_close_inr",
                "predicted_return_pct",
                "predicted_price_inr",
                "signal",
                "training_rows",
            ]
        ].copy()

        display["latest_close_inr"] = display["latest_close_inr"].map(
            lambda value: f"₹{value:,.2f}"
        )
        display["predicted_return_pct"] = display[
            "predicted_return_pct"
        ].map(
            lambda value: f"{value:+.2f}%"
        )
        display["predicted_price_inr"] = display[
            "predicted_price_inr"
        ].map(
            lambda value: f"₹{value:,.2f}"
        )

        print("\n" + "=" * 110)
        print(f"{horizon.upper()} PREDICTION")
        print("=" * 110)
        print(display.to_string(index=False))


def main() -> None:
    print("=" * 110)
    print("INDIAN STOCK TRADER — WEEKLY AND MONTHLY PREDICTIONS")
    print("=" * 110)

    db = PostgresDatabase()

    if not db.test_connection():
        raise ConnectionError(
            "PostgreSQL connection failed. Check values in your .env file."
        )

    db.create_schema()

    raw_data = db.get_market_data(symbols=SYMBOLS)

    if raw_data.empty:
        raise RuntimeError(
            "No OHLCV data in market_ohlcv. "
            "Run Yahoo Finance → PostgreSQL data ingestion first."
        )

    print(
        f"\nLoaded {len(raw_data):,} stored OHLCV rows "
        f"for {raw_data['symbol'].nunique()} symbols."
    )

    all_predictions = []

    for symbol in SYMBOLS:
        print(f"Training weekly and monthly models for {symbol}...")
        all_predictions.extend(
            predict_one_symbol(
                symbol=symbol,
                all_raw_data=raw_data,
            )
        )

    prediction_df = pd.DataFrame(all_predictions)

    if prediction_df.empty:
        print("\nNo forecasts created. Check model errors above.")
        return

    prediction_df.to_csv(
        "weekly_monthly_predictions.csv",
        index=False,
    )

    save_predictions_to_database(
        db=db,
        prediction_df=prediction_df,
    )

    print_predictions(prediction_df)

    print("\n" + "-" * 110)
    print("Saved CSV: weekly_monthly_predictions.csv")
    print("Saved PostgreSQL table: model_signals")
    print("-" * 110)

    print(
        "\nImportant: These are model outputs, not validated trade instructions. "
        "Evaluate each horizon through walk-forward backtesting before paper trading."
    )


if __name__ == "__main__":
    main()