"""
Entry-decision engine for the Indian Stock Trader project.

Converts raw weekly/monthly prediction signals into either:

    ELIGIBLE_BUY
    BLOCKED

It does NOT send broker orders.

Evaluation conditions:
1. Weekly forecast must be BUY.
2. Forecast must pass cost-and-error hurdle.
3. Monthly forecast must not be SELL / EXIT.
4. Close must be above SMA-50.
5. Nifty 50 / Nifty 500 regime must not be strongly bearish.
6. Average traded value must meet minimum liquidity.
7. Existing open position blocks duplicate entry.
8. Sector / portfolio exposure must permit entry.
9. ATR-based position size must be valid.
"""

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from database import PostgresDatabase
from feature_engineering import FeatureEngineer
from settings import config


# Windows terminals default to cp1252, which cannot encode symbols like
# the rupee sign (\u20b9). Force UTF-8 so console printing never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# You may use ^NSEI if it is in your Yahoo data/provider.
# A local broad-index series is preferred when you have it.
MARKET_SYMBOL_PRIORITY = [
    "^NSEI",
    "NIFTY50.NS",
    "NIFTY500.NS",
]

MODEL_NAME = "xgboost_direct_horizon"
MODEL_VERSION = "v1"


class EntryDecisionEngine:
    """Apply entry filters and position sizing to model predictions."""

    def __init__(
        self,
        db: PostgresDatabase,
        portfolio_value: float = 1_000_000,
    ):
        self.db = db
        self.portfolio_value = float(portfolio_value)

    def get_latest_predictions(self) -> pd.DataFrame:
        """
        Return the newest saved record per symbol and horizon.
        """
        from sqlalchemy import text

        query = text("""
            SELECT DISTINCT ON (symbol, prediction_horizon)
                symbol,
                signal_date,
                prediction_horizon,
                predicted_return,
                signal,
                model_name,
                model_version,
                created_at
            FROM model_signals
            WHERE model_name = :model_name
              AND model_version = :model_version
              AND prediction_horizon IN ('daily_1d','weekly_5d', 'monthly_20d')
            ORDER BY
                symbol,
                prediction_horizon,
                signal_date DESC,
                created_at DESC;
        """)

        with self.db.engine.connect() as connection:
            df = pd.read_sql(
                query,
                connection,
                params={
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                },
            )

        if df.empty:
            return df

        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["predicted_return"] = pd.to_numeric(
            df["predicted_return"],
            errors="coerce",
        )
        df["signal"] = pd.to_numeric(
            df["signal"],
            errors="coerce",
        ).fillna(0).astype(int)

        return df

    def get_active_positions(self) -> pd.DataFrame:
        """
        Current placeholder position source.

        Until broker integration is added, return an empty DataFrame.
        Replace this later with a portfolio_positions table populated
        by paper-trading or broker execution records.
        """
        return pd.DataFrame(
            columns=[
                "symbol",
                "quantity",
                "entry_price",
                "sector",
                "market_value",
            ]
        )

    def get_sector_for_symbol(self, symbol: str) -> str:
        """
        Read sector from your local Nifty 500 universe table where available.
        Falls back to UNKNOWN.
        """
        from sqlalchemy import text

        query = text("""
            SELECT COALESCE(industry, 'UNKNOWN') AS sector
            FROM nifty500_constituents
            WHERE yahoo_symbol = :symbol
              AND is_active = TRUE
            LIMIT 1;
        """)

        try:
            with self.db.engine.connect() as connection:
                row = connection.execute(
                    query,
                    {"symbol": symbol},
                ).fetchone()

            if row and row[0]:
                return str(row[0])

        except Exception as exc:
            logger.debug(
                "Sector lookup unavailable for %s: %s",
                symbol,
                exc,
            )

        # Small fallback mapping for your initial symbols.
        fallback = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Information Technology",
            "HDFCBANK.NS": "Banking",
            "INFY.NS": "Information Technology",
            "ICICIBANK.NS": "Banking",
        }

        return fallback.get(symbol, "UNKNOWN")

    def get_price_features(
        self,
        symbol: str,
    ) -> dict[str, Any] | None:
        """
        Read OHLCV data and compute current entry-filter features.
        """
        raw_df = self.db.get_market_data(symbols=[symbol])

        if raw_df.empty or len(raw_df) < 220:
            logger.warning(
                "%s: insufficient OHLCV data for SMA-200/ATR checks.",
                symbol,
            )
            return None

        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df = (
            raw_df
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        for column in ["open", "high", "low", "close", "volume"]:
            raw_df[column] = pd.to_numeric(
                raw_df[column],
                errors="coerce",
            ).astype("float64")

        # Use the existing feature module, which calculates SMA and ATR.
        engineer = FeatureEngineer()
        features_df = engineer.create_all_features(raw_df)

        if features_df.empty:
            return None

        latest = features_df.sort_values("date").tail(1).iloc[0]

        recent = raw_df.tail(config.LIQUIDITY_LOOKBACK_DAYS).copy()

        average_traded_value = (
            recent["close"] * recent["volume"]
        ).mean()

        return {
            "date": pd.Timestamp(latest["date"]),
            "close": float(latest["close"]),
            "sma_50": float(latest["sma_50"]),
            "sma_200": float(latest["sma_200"]),
            "atr_14": float(latest["atr_14"]),
            "average_traded_value_inr": float(average_traded_value),
        }

    def get_market_regime(self) -> dict[str, Any]:
        """
        Return Nifty market regime.

        This needs one market-index series in market_ohlcv.
        If none exists, it returns UNKNOWN and blocks entries by default.
        """
        for market_symbol in MARKET_SYMBOL_PRIORITY:
            raw_df = self.db.get_market_data(symbols=[market_symbol])

            if raw_df.empty or len(raw_df) < config.REGIME_SMA_LONG:
                continue

            raw_df = raw_df.sort_values("date").copy()
            raw_df["close"] = pd.to_numeric(
                raw_df["close"],
                errors="coerce",
            )

            raw_df["sma_50"] = raw_df["close"].rolling(
                config.REGIME_SMA_SHORT
            ).mean()

            raw_df["sma_200"] = raw_df["close"].rolling(
                config.REGIME_SMA_LONG
            ).mean()

            latest = raw_df.dropna(
                subset=["sma_50", "sma_200"]
            ).tail(1)

            if latest.empty:
                continue

            latest = latest.iloc[0]

            market_close = float(latest["close"])
            market_sma_50 = float(latest["sma_50"])
            market_sma_200 = float(latest["sma_200"])

            if market_close > market_sma_50 > market_sma_200:
                regime = "BULLISH"
            elif market_close > market_sma_200:
                regime = "NEUTRAL"
            else:
                regime = "BEARISH"

            return {
                "market_symbol": market_symbol,
                "market_close": market_close,
                "market_sma_50": market_sma_50,
                "market_sma_200": market_sma_200,
                "market_regime": regime,
            }

        return {
            "market_symbol": None,
            "market_close": None,
            "market_sma_50": None,
            "market_sma_200": None,
            "market_regime": "UNKNOWN",
        }

    def calculate_current_exposure(
        self,
        positions_df: pd.DataFrame,
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate total invested portfolio fraction and sector fractions.
        """
        if positions_df.empty:
            return 0.0, {}

        positions = positions_df.copy()

        if "market_value" not in positions.columns:
            positions["market_value"] = (
                positions["quantity"] * positions["entry_price"]
            )

        total_position_value = positions["market_value"].sum()
        total_exposure_pct = total_position_value / self.portfolio_value

        sector_values = (
            positions
            .groupby("sector", dropna=False)["market_value"]
            .sum()
            .to_dict()
        )

        sector_exposure_pct = {
            str(sector): value / self.portfolio_value
            for sector, value in sector_values.items()
        }

        return total_exposure_pct, sector_exposure_pct

    def calculate_position_size(
        self,
        entry_price: float,
        atr_14: float,
        current_portfolio_exposure_pct: float,
        current_sector_exposure_pct: float,
    ) -> dict[str, float | int]:
        """
        Position quantity constrained by:
        - INR risk permitted at initial ATR stop
        - per-stock notional cap
        - remaining portfolio exposure capacity
        - remaining sector exposure capacity
        """
        if entry_price <= 0 or atr_14 <= 0:
            return {
                "quantity": 0,
                "initial_stop_price": None,
                "risk_per_share": None,
                "position_value": 0.0,
            }

        initial_stop_price = (
            entry_price
            - config.ATR_STOP_MULTIPLIER * atr_14
        )

        risk_per_share = entry_price - initial_stop_price

        if risk_per_share <= 0:
            return {
                "quantity": 0,
                "initial_stop_price": None,
                "risk_per_share": None,
                "position_value": 0.0,
            }

        max_risk_inr = (
            self.portfolio_value
            * config.MAX_RISK_PER_TRADE_PCT
        )

        quantity_by_risk = int(
            max_risk_inr / risk_per_share
        )

        max_single_position_inr = (
            self.portfolio_value
            * config.MAX_POSITION_VALUE_PCT
        )

        remaining_portfolio_inr = max(
            0.0,
            (
                config.MAX_PORTFOLIO_EXPOSURE_PCT
                - current_portfolio_exposure_pct
            )
            * self.portfolio_value,
        )

        remaining_sector_inr = max(
            0.0,
            (
                config.MAX_SECTOR_EXPOSURE_PCT
                - current_sector_exposure_pct
            )
            * self.portfolio_value,
        )

        max_position_inr = min(
            max_single_position_inr,
            remaining_portfolio_inr,
            remaining_sector_inr,
        )

        quantity_by_capital = int(
            max_position_inr / entry_price
        )

        quantity = min(
            quantity_by_risk,
            quantity_by_capital,
        )

        if quantity < config.MIN_SHARES_PER_TRADE:
            quantity = 0

        return {
            "quantity": int(quantity),
            "initial_stop_price": float(initial_stop_price),
            "risk_per_share": float(risk_per_share),
            "position_value": float(quantity * entry_price),
        }

    def evaluate_symbol(
        self,
        symbol: str,
        daily_prediction: pd.Series | None,
        weekly_prediction: pd.Series | None,
        monthly_prediction: pd.Series | None,
        market_regime: dict[str, Any],
        active_positions: pd.DataFrame,
        current_portfolio_exposure_pct: float,
        sector_exposure_map: dict[str, float],
    ) -> dict[str, Any]:
        """Evaluate all nine entry conditions for one stock."""
        reasons = []

        daily_return = (
            float(daily_prediction["predicted_return"])
            if daily_prediction is not None
            else None
        )

        raw_daily_signal = (
            int(daily_prediction["signal"])
            if daily_prediction is not None
            else 0
        )

        weekly_return = (
            float(weekly_prediction["predicted_return"])
            if weekly_prediction is not None
            else None
        )

        monthly_return = (
            float(monthly_prediction["predicted_return"])
            if monthly_prediction is not None
            else None
        )

        raw_weekly_signal = (
            int(weekly_prediction["signal"])
            if weekly_prediction is not None
            else 0
        )

        raw_monthly_signal = (
            int(monthly_prediction["signal"])
            if monthly_prediction is not None
            else 0
        )

        missing_horizons = []

        if daily_prediction is None:
            missing_horizons.append("DAILY")

        if weekly_prediction is None:
            missing_horizons.append("WEEKLY")

        if monthly_prediction is None:
            missing_horizons.append("MONTHLY")

        latest_features = self.get_price_features(symbol)

        if latest_features is None:
            return {
                "symbol": symbol,
                "decision_date": pd.Timestamp.now().normalize(),
                "weekly_predicted_return": weekly_return,
                "monthly_predicted_return": monthly_return,
                "raw_weekly_signal": raw_weekly_signal,
                "raw_monthly_signal": raw_monthly_signal,
                "entry_status": "BLOCKED",
                "rejection_reasons": "INSUFFICIENT_PRICE_HISTORY",
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            }

        # 1. Weekly forecast is BUY.
        if raw_weekly_signal != 1:
            reasons.append("WEEKLY_SIGNAL_NOT_BUY")

        # 2. Forecast must clear cost + partial historical error buffer.
        required_forecast = max(
            config.WEEKLY_MIN_PREDICTED_RETURN,
            config.ESTIMATED_ROUND_TRIP_COST
            + (
                config.WEEKLY_ERROR_BUFFER
                * config.ERROR_BUFFER_MULTIPLIER
            ),
        )

        if weekly_return is None or weekly_return < required_forecast:
            reasons.append(
                "WEEKLY_RETURN_BELOW_COST_ERROR_BUFFER"
            )

        # 3. Monthly forecast must not be SELL / EXIT.
        if (
            monthly_prediction is None
            or raw_monthly_signal == -1
            or monthly_return is None
            or monthly_return < config.MONTHLY_MIN_PREDICTED_RETURN
        ):
            reasons.append("MONTHLY_FORECAST_BEARISH")

        # Daily signal is a timing gate, not the main weekly/monthly thesis.
        # If daily is SELL, a stock can wait for better immediate timing
        # instead of being rejected permanently.
        daily_timing_wait = (
            daily_prediction is not None
            and raw_daily_signal == -1
        )

        # 4. Stock above 50-day moving average.
        if (
            config.REQUIRE_CLOSE_ABOVE_SMA_50
            and latest_features["close"] <= latest_features["sma_50"]
        ):
            reasons.append("CLOSE_NOT_ABOVE_SMA_50")

        # 5. Market regime must not be strongly bearish.
        if market_regime["market_regime"] in {"BEARISH", "UNKNOWN"}:
            reasons.append(
                f"MARKET_REGIME_{market_regime['market_regime']}"
            )

        # 6. Average traded value must be high enough.
        if (
            latest_features["average_traded_value_inr"]
            < config.MIN_AVG_TRADED_VALUE_INR
        ):
            reasons.append("INSUFFICIENT_LIQUIDITY")

        # 7. Do not enter the same stock twice.
        already_held = (
            not active_positions.empty
            and symbol in active_positions["symbol"].astype(str).values
        )

        if already_held:
            reasons.append("OPEN_POSITION_ALREADY_EXISTS")

        # 8. Portfolio and sector limits.
        sector = self.get_sector_for_symbol(symbol)

        current_sector_exposure_pct = sector_exposure_map.get(
            sector,
            0.0,
        )

        if (
            current_portfolio_exposure_pct
            >= config.MAX_PORTFOLIO_EXPOSURE_PCT
        ):
            reasons.append("MAX_PORTFOLIO_EXPOSURE_REACHED")

        if (
            current_sector_exposure_pct
            >= config.MAX_SECTOR_EXPOSURE_PCT
        ):
            reasons.append("MAX_SECTOR_EXPOSURE_REACHED")

        if len(active_positions) >= config.MAX_OPEN_POSITIONS:
            reasons.append("MAX_OPEN_POSITIONS_REACHED")

        # 9. ATR-based position size must be valid.
        sizing = self.calculate_position_size(
            entry_price=latest_features["close"],
            atr_14=latest_features["atr_14"],
            current_portfolio_exposure_pct=current_portfolio_exposure_pct,
            current_sector_exposure_pct=current_sector_exposure_pct,
        )

        if sizing["quantity"] < config.MIN_SHARES_PER_TRADE:
            reasons.append("INVALID_POSITION_SIZE")

        if missing_horizons:
            status = "INSUFFICIENT_DATA"

            missing_text = "_".join(missing_horizons)
            reasons.insert(
                0,
                f"MISSING_{missing_text}_PREDICTION",
            )

        elif reasons:
            status = "BLOCKED"

        elif daily_timing_wait:
            status = "WAIT_DAILY_TIMING"

        else:
            status = "ELIGIBLE_BUY"

        return {
            "symbol": symbol,
            "decision_date": latest_features["date"],
            "daily_predicted_return": daily_return,
            "weekly_predicted_return": weekly_return,
            "monthly_predicted_return": monthly_return,
            "latest_close": latest_features["close"],
            "sma_50": latest_features["sma_50"],
            "atr_14": latest_features["atr_14"],
            "average_traded_value_inr": (
                latest_features["average_traded_value_inr"]
            ),
            "market_symbol": market_regime["market_symbol"],
            "market_close": market_regime["market_close"],
            "market_sma_50": market_regime["market_sma_50"],
            "market_sma_200": market_regime["market_sma_200"],
            "market_regime": market_regime["market_regime"],
            "raw_daily_signal": raw_daily_signal,
            "raw_weekly_signal": raw_weekly_signal,
            "raw_monthly_signal": raw_monthly_signal,
            "entry_status": status,
            "rejection_reasons": "; ".join(reasons) if reasons else None,
            "entry_price": latest_features["close"],
            "initial_stop_price": sizing["initial_stop_price"],
            "risk_per_share": sizing["risk_per_share"],
            "suggested_quantity": sizing["quantity"],
            "suggested_position_value": sizing["position_value"],
            "current_portfolio_value": self.portfolio_value,
            "current_portfolio_exposure_pct": (
                current_portfolio_exposure_pct
            ),
            "sector": sector,
            "sector_exposure_pct": current_sector_exposure_pct,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

    def run(self) -> pd.DataFrame:
        """
        Evaluate all symbols that have latest weekly/monthly model data.
        """
        predictions = self.get_latest_predictions()

        if predictions.empty:
            raise RuntimeError(
                "No weekly/monthly predictions found in model_signals. "
                "Run predict_weekly_monthly.py first."
            )

        symbols = sorted(predictions["symbol"].unique())

        active_positions = self.get_active_positions()

        (
            current_portfolio_exposure_pct,
            sector_exposure_map,
        ) = self.calculate_current_exposure(active_positions)

        market_regime = self.get_market_regime()

        logger.info(
            "Market regime: %s (%s)",
            market_regime["market_regime"],
            market_regime["market_symbol"],
        )

        decisions = []

        for symbol in symbols:
            symbol_predictions = predictions[
                predictions["symbol"] == symbol
            ].copy()

            daily = symbol_predictions[
                symbol_predictions["prediction_horizon"] == "daily_1d"
            ]
            
            weekly = symbol_predictions[
                symbol_predictions["prediction_horizon"] == "weekly_5d"
            ]

            monthly = symbol_predictions[
                symbol_predictions["prediction_horizon"] == "monthly_20d"
            ]

            daily_row = daily.iloc[0] if not daily.empty else None
            weekly_row = weekly.iloc[0] if not weekly.empty else None
            monthly_row = monthly.iloc[0] if not monthly.empty else None

            decision = self.evaluate_symbol(
                symbol=symbol,
                daily_prediction=daily_row,
                weekly_prediction=weekly_row,
                monthly_prediction=monthly_row,
                market_regime=market_regime,
                active_positions=active_positions,
                current_portfolio_exposure_pct=(
                    current_portfolio_exposure_pct
                ),
                sector_exposure_map=sector_exposure_map,
            )

            decisions.append(decision)

        decisions_df = pd.DataFrame(decisions)

        self.db.save_entry_decisions(decisions_df)

        return decisions_df


def print_summary(decisions_df: pd.DataFrame) -> None:
    """Print eligible and blocked entries clearly."""
    if decisions_df.empty:
        print("No entry decisions generated.")
        return

    display_columns = [
        "symbol",
        "entry_status",
        "daily_predicted_return",
        "weekly_predicted_return",
        "monthly_predicted_return",
        "latest_close",
        "sma_50",
        "atr_14",
        "average_traded_value_inr",
        "market_regime",
        "sector",
        "suggested_quantity",
        "initial_stop_price",
        "suggested_position_value",
        "rejection_reasons",
    ]

    output = decisions_df.copy()

    for column in [
        "daily_predicted_return",
        "weekly_predicted_return",
        "monthly_predicted_return",
    ]:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce") * 100
        ).map(
            lambda value: f"{value:+.2f}%"
            if pd.notna(value)
            else "—"
        )

    for column in [
        "latest_close",
        "sma_50",
        "atr_14",
        "average_traded_value_inr",
        "initial_stop_price",
        "suggested_position_value",
    ]:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).map(
            lambda value: f"₹{value:,.2f}"
            if pd.notna(value)
            else "—"
        )

    print("\n" + "=" * 180)
    print("ENTRY DECISION RESULTS")
    print("=" * 180)
    print(
        output[display_columns]
        .sort_values(
            ["entry_status", "symbol"],
            ascending=[False, True],
        )
        .to_string(index=False)
    )

    print("\n" + "-" * 180)
    print(
        output["entry_status"]
        .value_counts()
        .rename_axis("Status")
        .to_string()
    )
    print("-" * 180)


def main() -> None:
    db = PostgresDatabase()

    if not db.test_connection():
        raise ConnectionError(
            "Could not connect to PostgreSQL. Check your .env file."
        )

    db.create_schema()

    engine = EntryDecisionEngine(
        db=db,
        portfolio_value=config.INITIAL_CAPITAL,
    )

    decisions_df = engine.run()

    decisions_df.to_csv(
        "entry_decision_report.csv",
        index=False,
    )

    print_summary(decisions_df)

    print(
        "\nSaved local report: entry_decision_report.csv"
    )
    print(
        "Saved database table: entry_decisions"
    )
    print(
        "\nImportant: This engine produces paper-trading eligibility "
        "decisions only. It does not place broker orders."
    )


if __name__ == "__main__":
    main()