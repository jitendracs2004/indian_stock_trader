"""
Streamlit dashboard for Indian Stock Trader.

Reads local PostgreSQL data:
- market_ohlcv: historical daily NSE/BSE price data
- model_signals: daily, weekly, monthly model predictions

Run:
    streamlit run streamlit_dashboard.py
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from database import PostgresDatabase


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Stock Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SIGNAL_MAP = {
    1: "BUY",
    0: "HOLD",
    -1: "SELL / EXIT",
}

SIGNAL_COLOR_MAP = {
    1: "#16a34a",      # green
    0: "#f59e0b",      # amber
    -1: "#dc2626",     # red
}

HORIZON_CONFIG = {
    "daily_1d": {
        "label": "Daily",
        "sub_label": "1 trading day",
        "days": 1,
    },
    "weekly_5d": {
        "label": "Weekly",
        "sub_label": "5 trading days",
        "days": 5,
    },
    "monthly_20d": {
        "label": "Monthly",
        "sub_label": "20 trading days",
        "days": 20,
    },
}


# ---------------------------------------------------------------------
# Database functions
# ---------------------------------------------------------------------
@st.cache_resource
def get_database() -> PostgresDatabase:
    """Create one reusable PostgreSQL connection manager."""
    db = PostgresDatabase()

    if not db.test_connection():
        raise ConnectionError(
            "PostgreSQL connection failed. "
            "Check your .env file and ensure PostgreSQL is running."
        )

    return db


@st.cache_data(ttl=60)
def load_symbols() -> list[str]:
    """Load symbols that exist in market_ohlcv."""
    db = get_database()

    query = text("""
        SELECT DISTINCT symbol
        FROM market_ohlcv
        ORDER BY symbol;
    """)

    with db.engine.connect() as connection:
        rows = connection.execute(query).fetchall()

    return [row[0] for row in rows]


@st.cache_data(ttl=60)
def load_price_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Load recent OHLCV history for one symbol."""
    db = get_database()

    query = text("""
        SELECT
            trade_date AS date,
            open,
            high,
            low,
            close,
            volume
        FROM market_ohlcv
        WHERE symbol = :symbol
        ORDER BY trade_date DESC
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "symbol": symbol,
                "limit_rows": days,
            },
        )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


@st.cache_data(ttl=60)
def load_latest_predictions(symbol: str) -> pd.DataFrame:
    """
    Load latest model prediction by horizon for one NSE symbol.
    """
    db = get_database()

    query = text("""
        SELECT DISTINCT ON (prediction_horizon)
            symbol,
            signal_date,
            prediction_horizon,
            predicted_return,
            signal,
            COALESCE(model_name, 'unknown_model') AS model_name,
            COALESCE(model_version, 'v1') AS model_version,
            COALESCE(created_at, NOW()) AS created_at
        FROM model_signals
        WHERE symbol = :symbol
        ORDER BY
            prediction_horizon,
            signal_date DESC,
            created_at DESC NULLS LAST;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"symbol": symbol},
        )

    if df.empty:
        return df

    df["signal_date"] = pd.to_datetime(
        df["signal_date"],
        errors="coerce",
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    df["signal"] = pd.to_numeric(
        df["signal"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["signal_label"] = df["signal"].map(
        SIGNAL_MAP
    ).fillna("UNKNOWN")

    df["predicted_return_pct"] = (
        pd.to_numeric(
            df["predicted_return"],
            errors="coerce",
        ).fillna(0.0) * 100
    )

    return df

@st.cache_data(ttl=60)
def load_latest_entry_decision(
    symbol: str,
) -> pd.DataFrame:
    """
    Load the newest final entry decision for a selected symbol.

    Supports Phase 2 fields:
    - daily_predicted_return
    - weekly_predicted_return
    - monthly_predicted_return
    - raw_daily_signal
    - raw_weekly_signal
    - raw_monthly_signal
    - entry_status
    """
    db = get_database()

    query = text("""
        SELECT
            id,
            symbol,
            decision_date,

            daily_predicted_return,
            weekly_predicted_return,
            monthly_predicted_return,

            latest_close,
            sma_50,
            atr_14,
            average_traded_value_inr,

            market_symbol,
            market_close,
            market_sma_50,
            market_sma_200,
            market_regime,

            raw_daily_signal,
            raw_weekly_signal,
            raw_monthly_signal,

            entry_status,
            rejection_reasons,

            entry_price,
            initial_stop_price,
            risk_per_share,
            suggested_quantity,
            suggested_position_value,

            current_portfolio_value,
            current_portfolio_exposure_pct,
            sector,
            sector_exposure_pct,

            model_name,
            model_version,
            created_at
        FROM entry_decisions
        WHERE symbol = :symbol
        ORDER BY
            decision_date DESC,
            created_at DESC,
            id DESC
        LIMIT 1;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"symbol": symbol},
        )

    if df.empty:
        return df

    date_columns = [
        "decision_date",
        "created_at",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    numeric_columns = [
        "daily_predicted_return",
        "weekly_predicted_return",
        "monthly_predicted_return",
        "latest_close",
        "sma_50",
        "atr_14",
        "average_traded_value_inr",
        "market_close",
        "market_sma_50",
        "market_sma_200",
        "entry_price",
        "initial_stop_price",
        "risk_per_share",
        "suggested_position_value",
        "current_portfolio_value",
        "current_portfolio_exposure_pct",
        "sector_exposure_pct",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    integer_columns = [
        "raw_daily_signal",
        "raw_weekly_signal",
        "raw_monthly_signal",
        "suggested_quantity",
    ]

    for column in integer_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0).astype(int)

    df["entry_status"] = df["entry_status"].fillna(
        "INSUFFICIENT_DATA"
    )

    df["rejection_reasons"] = df[
        "rejection_reasons"
    ].fillna("")

    return df

@st.cache_data(ttl=60)
def load_top_eligible_buys(
    limit_rows: int = 10,
) -> pd.DataFrame:
    """
    Load the Top 10 latest final entry decisions that passed every rule.

    Important:
    - Only entry_status = ELIGIBLE_BUY is shown.
    - Returns one newest decision per stock.
    - Ranked by weekly model forecast, then monthly forecast.
    """
    db = get_database()

    query = text("""
        WITH latest_decision_per_symbol AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                decision_date,
                weekly_predicted_return,
                monthly_predicted_return,
                latest_close,
                sma_50,
                atr_14,
                average_traded_value_inr,
                market_symbol,
                market_regime,
                entry_status,
                rejection_reasons,
                entry_price,
                initial_stop_price,
                risk_per_share,
                suggested_quantity,
                suggested_position_value,
                current_portfolio_value,
                current_portfolio_exposure_pct,
                sector,
                sector_exposure_pct,
                model_name,
                model_version,
                created_at
            FROM entry_decisions
            ORDER BY
                symbol,
                decision_date DESC,
                created_at DESC
        )
        SELECT *
        FROM latest_decision_per_symbol
        WHERE entry_status = 'ELIGIBLE_BUY'
        ORDER BY
            weekly_predicted_return DESC NULLS LAST,
            monthly_predicted_return DESC NULLS LAST
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"limit_rows": limit_rows},
        )

    if df.empty:
        return df

    df["decision_date"] = pd.to_datetime(
        df["decision_date"],
        errors="coerce",
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    numeric_cols = [
        "weekly_predicted_return",
        "monthly_predicted_return",
        "latest_close",
        "sma_50",
        "atr_14",
        "average_traded_value_inr",
        "entry_price",
        "initial_stop_price",
        "risk_per_share",
        "suggested_position_value",
        "current_portfolio_value",
        "current_portfolio_exposure_pct",
        "sector_exposure_pct",
    ]

    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["suggested_quantity"] = pd.to_numeric(
        df["suggested_quantity"],
        errors="coerce",
    ).fillna(0).astype(int)

    return df

@st.cache_data(ttl=60)
def load_top_buy_predictions(
    horizon: str,
    limit_rows: int = 10,
) -> pd.DataFrame:
    """
    Return top BUY predictions for one horizon.

    Logic:
    1. Select only the newest prediction for each symbol.
    2. Keep only numeric BUY signal = 1.
    3. Rank highest predicted returns first.
    """
    db = get_database()

    query = text("""
        WITH latest_per_symbol AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                signal_date,
                prediction_horizon,
                predicted_return,
                signal,
                model_name,
                model_version,
                created_at
            FROM model_signals
            WHERE prediction_horizon = :horizon
            ORDER BY
                symbol,
                signal_date DESC,
                created_at DESC
        )
        SELECT
            symbol,
            signal_date,
            prediction_horizon,
            predicted_return,
            signal,
            model_name,
            model_version,
            created_at
        FROM latest_per_symbol
        WHERE signal = 1
        ORDER BY predicted_return DESC
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "horizon": horizon,
                "limit_rows": limit_rows,
            },
        )

    if not df.empty:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df["created_at"] = pd.to_datetime(df["created_at"])

        df["predicted_return"] = pd.to_numeric(
            df["predicted_return"],
            errors="coerce",
        )

        df["predicted_return_pct"] = (
            df["predicted_return"] * 100
        )

        df["signal"] = pd.to_numeric(
            df["signal"],
            errors="coerce",
        ).fillna(0).astype(int)

        df["signal_label"] = df["signal"].map(SIGNAL_MAP)

    return df

def show_top_buy_table(
    horizon: str,
    latest_prices_df: pd.DataFrame,
) -> None:
    """
    Display the Top 10 latest BUY recommendations for a horizon.

    latest_prices_df must contain:
    symbol, close
    """
    horizon_info = HORIZON_CONFIG[horizon]

    top_buy_df = load_top_buy_predictions(
        horizon=horizon,
        limit_rows=10,
    )

    st.subheader(
        f"Top 10 BUY — {horizon_info['label']} "
        f"({horizon_info['sub_label']})"
    )

    if top_buy_df.empty:
        st.info(
            f"No active BUY signals currently exist for "
            f"{horizon_info['label'].lower()} predictions."
        )
        return

    prices = latest_prices_df[
        ["symbol", "close"]
    ].copy()

    prices = prices.rename(
        columns={"close": "latest_close_inr"}
    )

    display_df = top_buy_df.merge(
        prices,
        on="symbol",
        how="left",
    )

    display_df["expected_price_inr"] = (
        display_df["latest_close_inr"]
        * (1 + display_df["predicted_return"])
    )

    display_df = display_df.sort_values(
        "predicted_return",
        ascending=False,
    ).reset_index(drop=True)

    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    display_df["signal_date"] = display_df[
        "signal_date"
    ].dt.strftime("%d %b %Y")

    display_df["latest_close_inr"] = display_df[
        "latest_close_inr"
    ].map(format_inr)

    display_df["expected_price_inr"] = display_df[
        "expected_price_inr"
    ].map(format_inr)

    display_df["predicted_return_pct"] = display_df[
        "predicted_return_pct"
    ].map(format_pct)

    display_df = display_df[
        [
            "symbol",
            "signal_date",
            "latest_close_inr",
            "predicted_return_pct",
            "expected_price_inr",
            "model_name",
            "model_version",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "signal_date": "Prediction Date",
            "latest_close_inr": "Latest Close",
            "predicted_return_pct": "Predicted Return",
            "expected_price_inr": "Expected Price",
            "model_name": "Model",
            "model_version": "Version",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
    )

    st.caption(
        "Ranked by model-predicted return among the latest active "
        "BUY signals. This is a research ranking, not financial advice."
    )

def show_top_eligible_buys() -> None:
    """
    Show only stocks that passed all nine entry checks.

    This is the primary Top 10 shortlist for paper trading.
    """
    eligible_df = load_top_eligible_buys(limit_rows=10)

    st.header("🏆 Top 10 Eligible BUY Candidates")

    st.caption(
        "These stocks passed the final entry engine: weekly forecast, "
        "cost/error buffer, monthly confirmation, SMA-50 trend, market "
        "regime, liquidity, duplicate-position check, exposure limits, "
        "and ATR-based position sizing."
    )

    if eligible_df.empty:
        st.warning(
            "No stocks currently qualify as ELIGIBLE_BUY. "
            "This can be normal when the broad market is bearish, "
            "signals are weak, or liquidity/trend filters block entries."
        )

        st.info(
            "Run these jobs in order after market data is refreshed:\n\n"
            "1. `python predict_weekly_monthly.py`\n\n"
            "2. `python entry_decision_engine.py`\n\n"
            "3. Refresh this dashboard."
        )
        return

    display_df = eligible_df.copy()

    display_df["weekly_return_pct"] = (
        display_df["weekly_predicted_return"] * 100
    )

    display_df["monthly_return_pct"] = (
        display_df["monthly_predicted_return"] * 100
    )

    display_df["stop_distance_pct"] = (
        (
            display_df["entry_price"]
            - display_df["initial_stop_price"]
        )
        / display_df["entry_price"]
        * 100
    )

    display_df = display_df.sort_values(
        by=[
            "weekly_predicted_return",
            "monthly_predicted_return",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    display_df["decision_date"] = display_df[
        "decision_date"
    ].dt.strftime("%d %b %Y")

    for column in [
        "latest_close",
        "entry_price",
        "initial_stop_price",
        "suggested_position_value",
        "average_traded_value_inr",
    ]:
        display_df[column] = display_df[column].map(
            format_inr
        )

    for column in [
        "weekly_return_pct",
        "monthly_return_pct",
        "stop_distance_pct",
    ]:
        display_df[column] = display_df[column].map(
            format_pct
        )

    display_df["suggested_quantity"] = display_df[
        "suggested_quantity"
    ].map(
        lambda value: f"{int(value):,}"
    )

    display_df["market_regime"] = display_df[
        "market_regime"
    ].fillna("UNKNOWN")

    display_df["sector"] = display_df[
        "sector"
    ].fillna("UNKNOWN")

    display_df = display_df[
        [
            "symbol",
            "decision_date",
            "sector",
            "market_regime",
            "latest_close",
            "weekly_return_pct",
            "monthly_return_pct",
            "entry_price",
            "initial_stop_price",
            "stop_distance_pct",
            "suggested_quantity",
            "suggested_position_value",
            "average_traded_value_inr",
            "model_name",
            "model_version",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "decision_date": "Decision Date",
            "sector": "Sector",
            "market_regime": "Market Regime",
            "latest_close": "Latest Close",
            "weekly_return_pct": "Weekly Forecast",
            "monthly_return_pct": "Monthly Forecast",
            "entry_price": "Suggested Entry",
            "initial_stop_price": "Initial Stop",
            "stop_distance_pct": "Stop Distance",
            "suggested_quantity": "Suggested Qty",
            "suggested_position_value": "Position Value",
            "average_traded_value_inr": "20D Avg Traded Value",
            "model_name": "Model",
            "model_version": "Version",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        height=420,
    )

    st.caption(
        "Suggested Entry is the latest available close used by the "
        "research engine—not a guaranteed executable price. Initial Stop "
        "is based on ATR volatility. The suggested quantity is a "
        "paper-trading risk allocation, not a broker order."
    )

@st.cache_data(ttl=60)
def load_waiting_daily_timing_candidates(
    limit_rows: int = 10,
) -> pd.DataFrame:
    """
    Load latest candidate decisions where weekly/monthly logic passed,
    but daily timing is currently SELL / EXIT.
    """
    db = get_database()

    query = text("""
        WITH latest_per_symbol AS (
            SELECT DISTINCT ON (symbol)
                id,
                symbol,
                decision_date,
                daily_predicted_return,
                weekly_predicted_return,
                monthly_predicted_return,
                latest_close,
                entry_price,
                initial_stop_price,
                suggested_quantity,
                suggested_position_value,
                market_regime,
                sector,
                entry_status,
                rejection_reasons,
                created_at
            FROM entry_decisions
            ORDER BY
                symbol,
                decision_date DESC,
                created_at DESC,
                id DESC
        )
        SELECT *
        FROM latest_per_symbol
        WHERE entry_status = 'WAIT_DAILY_TIMING'
        ORDER BY
            weekly_predicted_return DESC NULLS LAST,
            monthly_predicted_return DESC NULLS LAST
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"limit_rows": limit_rows},
        )

    if df.empty:
        return df

    for column in [
        "daily_predicted_return",
        "weekly_predicted_return",
        "monthly_predicted_return",
        "latest_close",
        "entry_price",
        "initial_stop_price",
        "suggested_position_value",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["suggested_quantity"] = pd.to_numeric(
        df["suggested_quantity"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["decision_date"] = pd.to_datetime(
        df["decision_date"],
        errors="coerce",
    )

    return df

def show_wait_daily_timing_table() -> None:
    """
    Show stocks whose weekly/monthly trade thesis passed but whose
    daily timing currently says wait.
    """
    waiting_df = load_waiting_daily_timing_candidates(
        limit_rows=10
    )

    st.subheader("⏳ Top Candidates Waiting for Daily Timing")

    st.caption(
        "These stocks are not active entry candidates today. Their weekly "
        "and monthly conditions passed, but the daily model is bearish. "
        "They will be reconsidered after the next daily pipeline run."
    )

    if waiting_df.empty:
        st.info(
            "No symbols are currently waiting for daily timing."
        )
        return

    display_df = waiting_df.copy()

    for source_column, target_column in [
        ("daily_predicted_return", "Daily Forecast"),
        ("weekly_predicted_return", "Weekly Forecast"),
        ("monthly_predicted_return", "Monthly Forecast"),
    ]:
        display_df[target_column] = (
            display_df[source_column] * 100
        ).map(format_pct)

    display_df["Entry Reference"] = display_df[
        "entry_price"
    ].map(format_inr)

    display_df["Initial Stop"] = display_df[
        "initial_stop_price"
    ].map(format_inr)

    display_df["Decision Date"] = display_df[
        "decision_date"
    ].dt.strftime("%d %b %Y")

    display_df["Suggested Qty"] = display_df[
        "suggested_quantity"
    ].map(lambda value: f"{int(value):,}")

    display_df = display_df[
        [
            "symbol",
            "Decision Date",
            "sector",
            "market_regime",
            "Daily Forecast",
            "Weekly Forecast",
            "Monthly Forecast",
            "Entry Reference",
            "Initial Stop",
            "Suggested Qty",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
            "market_regime": "Market Regime",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=360,
    )

@st.cache_data(ttl=60)
def load_latest_prices() -> pd.DataFrame:
    """
    Load most recent close for every symbol from market_ohlcv.
    """
    db = get_database()

    query = text("""
        SELECT DISTINCT ON (symbol)
            symbol,
            trade_date AS date,
            close
        FROM market_ohlcv
        ORDER BY symbol, trade_date DESC;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(query, connection)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

    return df

@st.cache_data(ttl=60)
def load_prediction_history(
    symbol: str,
    limit_rows: int = 100,
) -> pd.DataFrame:
    """
    Load stored prediction history for one symbol.

    Uses NULL defaults for optional fields so the dashboard works with
    older versions of the model_signals PostgreSQL table too.
    """
    db = get_database()

    query = text("""
        SELECT
            symbol,
            signal_date,
            prediction_horizon,
            predicted_return,
            signal,
            COALESCE(model_name, 'unknown_model') AS model_name,
            COALESCE(model_version, 'v1') AS model_version,
            COALESCE(created_at, NOW()) AS created_at
        FROM model_signals
        WHERE symbol = :symbol
        ORDER BY
            signal_date DESC,
            created_at DESC NULLS LAST
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "symbol": symbol,
                "limit_rows": limit_rows,
            },
        )

    if df.empty:
        return df

    df["signal_date"] = pd.to_datetime(
        df["signal_date"],
        errors="coerce",
    )

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    df["signal"] = pd.to_numeric(
        df["signal"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["signal_label"] = df["signal"].map(
        SIGNAL_MAP
    ).fillna("UNKNOWN")

    df["predicted_return_pct"] = (
        pd.to_numeric(
            df["predicted_return"],
            errors="coerce",
        ).fillna(0.0) * 100
    )

    return df

@st.cache_data(ttl=60)
def load_portfolio_snapshot() -> pd.DataFrame:
    """Load the latest paper portfolio snapshot."""
    db = get_database()

    query = text("""
        SELECT
            portfolio_name,
            snapshot_date,
            cash_inr,
            holdings_value_inr,
            total_value_inr,
            open_positions,
            gross_exposure_pct,
            daily_pnl_inr,
            total_pnl_inr,
            drawdown_pct,
            created_at
        FROM portfolio_daily_snapshots
        WHERE portfolio_name = 'paper_default'
        ORDER BY snapshot_date DESC, created_at DESC
        LIMIT 1;
    """)

    with db.engine.connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=60)
def load_open_paper_positions() -> pd.DataFrame:
    """Load current open paper positions."""
    db = get_database()

    query = text("""
        SELECT
            id,
            symbol,
            sector,
            entry_date,
            entry_price,
            quantity,
            initial_stop_price,
            trailing_stop_price,
            highest_price_since_entry,
            current_price,
            market_value,
            unrealized_pnl_inr,
            unrealized_pnl_pct,
            entry_decision_id
        FROM portfolio_positions
        WHERE portfolio_name = 'paper_default'
          AND status = 'OPEN'
        ORDER BY unrealized_pnl_pct DESC NULLS LAST, entry_date ASC;
    """)

    with db.engine.connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=60)
def load_closed_paper_positions(limit_rows: int = 50) -> pd.DataFrame:
    """Load recently closed paper positions."""
    db = get_database()

    query = text("""
        SELECT
            symbol,
            sector,
            entry_date,
            entry_price,
            quantity,
            exit_date,
            exit_price,
            realized_pnl_inr,
            realized_pnl_pct,
            exit_reason
        FROM portfolio_positions
        WHERE portfolio_name = 'paper_default'
          AND status = 'CLOSED'
        ORDER BY exit_date DESC NULLS LAST
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
            params={"limit_rows": limit_rows},
        )


@st.cache_data(ttl=60)
def load_portfolio_history(days: int = 180) -> pd.DataFrame:
    """Load daily portfolio value history."""
    db = get_database()

    query = text("""
        SELECT
            snapshot_date,
            cash_inr,
            holdings_value_inr,
            total_value_inr,
            open_positions,
            gross_exposure_pct,
            daily_pnl_inr,
            total_pnl_inr,
            drawdown_pct
        FROM portfolio_daily_snapshots
        WHERE portfolio_name = 'paper_default'
        ORDER BY snapshot_date DESC
        LIMIT :days;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"days": days},
        )

    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df = df.sort_values("snapshot_date")

    return df

# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
def format_inr(value: float | None) -> str:
    """Format a numeric value in INR."""
    if value is None or pd.isna(value):
        return "—"

    return f"₹{value:,.2f}"

def format_pct(value: float | None) -> str:
    """
    Format a percentage value already expressed as a percentage.

    Examples:
        1.25  -> +1.25%
       -0.75  -> -0.75%
        0.00  -> +0.00%
        None  -> —
    """
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):+.2f}%"

def decision_status_display(status: str) -> tuple[str, str]:
    """
    Convert database entry_status into:
    - readable label
    - Streamlit message type

    Returns:
        (label, message_type)

    message_type is one of:
        success, warning, error, info
    """
    status_map = {
        "ELIGIBLE_BUY": (
            "✅ ELIGIBLE BUY",
            "success",
        ),
        "WAIT_DAILY_TIMING": (
            "⏳ WAIT FOR DAILY TIMING",
            "warning",
        ),
        "BLOCKED": (
            "⛔ ENTRY BLOCKED",
            "error",
        ),
        "INSUFFICIENT_DATA": (
            "⚪ INSUFFICIENT DATA",
            "info",
        ),
        "STALE_DATA": (
            "⚪ STALE DATA",
            "info",
        ),
    }

    return status_map.get(
        str(status),
        (f"⚪ {status}", "info"),
    )


def raw_signal_label(signal_code: int) -> str:
    """
    Convert stored numeric model signal to label.

     1 = BUY
     0 = HOLD
    -1 = SELL / EXIT
    """
    labels = {
        1: "BUY",
        0: "HOLD",
        -1: "SELL / EXIT",
    }

    return labels.get(int(signal_code), "UNKNOWN")


def display_status_message(
    status: str,
    symbol: str,
    rejection_reasons: str = "",
) -> None:
    """Render a clear final decision banner for one symbol."""
    label, message_type = decision_status_display(status)

    message = f"**{symbol} — {label}**"

    if status == "ELIGIBLE_BUY":
        message += (
            "  \nThe stock passed the weekly/monthly forecast, trend, "
            "market-regime, liquidity, portfolio, and position-sizing checks."
        )

    elif status == "WAIT_DAILY_TIMING":
        message += (
            "  \nWeekly and monthly conditions passed, but the daily "
            "forecast is currently bearish. Wait for the next daily "
            "pipeline run before considering an entry."
        )

    elif status == "BLOCKED":
        message += (
            "  \nA hard entry condition failed. No new paper position "
            "should be opened."
        )

    elif status == "INSUFFICIENT_DATA":
        message += (
            "  \nOne or more required prediction horizons or data inputs "
            "are missing."
        )

    if message_type == "success":
        st.success(message)

    elif message_type == "warning":
        st.warning(message)

    elif message_type == "error":
        st.error(message)

    else:
        st.info(message)

    if rejection_reasons and str(rejection_reasons).strip():
        st.caption(
            f"Decision detail: {rejection_reasons}"
        )


def get_prediction_row(
    predictions_df: pd.DataFrame,
    horizon: str,
) -> pd.Series | None:
    """Get the current row for a given daily/weekly/monthly horizon."""
    if predictions_df.empty:
        return None

    filtered = predictions_df[
        predictions_df["prediction_horizon"] == horizon
    ]

    if filtered.empty:
        return None

    return filtered.iloc[0]


def create_price_chart(
    price_df: pd.DataFrame,
    symbol: str,
    chart_days: int,
) -> go.Figure:
    """Create a candlestick chart with 20/50-day moving averages."""
    chart_df = price_df.tail(chart_days).copy()

    chart_df["sma_20"] = chart_df["close"].rolling(20).mean()
    chart_df["sma_50"] = chart_df["close"].rolling(50).mean()

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["sma_20"],
            name="SMA 20",
            line=dict(color="#2563eb", width=1.5),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["sma_50"],
            name="SMA 50",
            line=dict(color="#9333ea", width=1.5),
        )
    )

    figure.update_layout(
        title=f"{symbol} — Daily Price Chart",
        height=550,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (INR)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return figure


def show_prediction_card(
    horizon: str,
    row: pd.Series | None,
    latest_close: float | None,
) -> None:
    """Render one daily/weekly/monthly prediction card."""
    config = HORIZON_CONFIG[horizon]

    with st.container(border=True):
        st.subheader(config["label"])
        st.caption(config["sub_label"])

        if row is None:
            st.info(
                f"No {config['label'].lower()} prediction is stored yet."
            )
            st.caption(
                "Generate and save this horizon in the prediction script."
            )
            return

        predicted_return_pct = float(row["predicted_return_pct"])
        signal_code = int(row["signal"])
        signal_label = row["signal_label"]

        expected_price = None
        if latest_close is not None:
            expected_price = latest_close * (
                1 + (predicted_return_pct / 100)
            )

        st.metric(
            label="Predicted return",
            value=format_pct(predicted_return_pct),
            delta=format_pct(predicted_return_pct),
        )

        st.metric(
            label="Expected price",
            value=format_inr(expected_price),
        )

        signal_color = SIGNAL_COLOR_MAP.get(signal_code, "#6b7280")

        st.markdown(
            f"""
            <div style="
                display: inline-block;
                color: white;
                background-color: {signal_color};
                padding: 7px 12px;
                border-radius: 8px;
                font-weight: 700;
                margin-top: 6px;
                margin-bottom: 6px;
            ">
                {signal_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"Prediction date: {row['signal_date'].strftime('%d %b %Y')}"
        )
        st.caption(
            f"Model: {row['model_name']} ({row['model_version']})"
        )


def create_return_comparison_chart(
    predictions_df: pd.DataFrame,
) -> go.Figure | None:
    """Create a bar chart of predicted returns by horizon."""
    if predictions_df.empty:
        return None

    chart_df = predictions_df.copy()

    chart_df["horizon_label"] = chart_df[
        "prediction_horizon"
    ].map(
        lambda horizon: HORIZON_CONFIG.get(
            horizon,
            {"label": horizon},
        )["label"]
    )

    chart_df["bar_color"] = chart_df["signal"].map(
        SIGNAL_COLOR_MAP
    ).fillna("#6b7280")

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_df["horizon_label"],
            y=chart_df["predicted_return_pct"],
            marker_color=chart_df["bar_color"],
            text=chart_df["predicted_return_pct"].map(
                lambda value: f"{value:+.2f}%"
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Predicted return: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_color="#6b7280",
    )

    figure.update_layout(
        title="Predicted Return by Horizon",
        template="plotly_white",
        height=350,
        yaxis_title="Predicted return (%)",
        xaxis_title="Prediction horizon",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return figure

def render_paper_portfolio() -> None:
    """Render the paper portfolio page."""
    st.title("💼 Paper Portfolio")
    st.caption(
        "Research-only portfolio tracking. No live broker orders are sent."
    )

    snapshot_df = load_portfolio_snapshot()
    open_positions_df = load_open_paper_positions()
    closed_positions_df = load_closed_paper_positions()
    history_df = load_portfolio_history()

    if snapshot_df.empty:
        st.info(
            "No paper portfolio snapshot exists yet. Run:\n\n"
            "`python portfolio_engine.py`"
        )
        return

    snapshot = snapshot_df.iloc[0]

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

    metric_1.metric(
        "Portfolio Value",
        format_inr(snapshot["total_value_inr"]),
        format_inr(snapshot["daily_pnl_inr"]),
    )

    metric_2.metric(
        "Cash Available",
        format_inr(snapshot["cash_inr"]),
    )

    metric_3.metric(
        "Holdings Value",
        format_inr(snapshot["holdings_value_inr"]),
    )

    metric_4.metric(
        "Open Positions",
        int(snapshot["open_positions"]),
    )

    metric_5.metric(
        "Drawdown",
        format_pct(snapshot["drawdown_pct"]),
    )

    if not history_df.empty:
        import plotly.graph_objects as go

        value_chart = go.Figure()

        value_chart.add_trace(
            go.Scatter(
                x=history_df["snapshot_date"],
                y=history_df["total_value_inr"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#2563eb", width=3),
            )
        )

        value_chart.update_layout(
            title="Paper Portfolio Value",
            template="plotly_white",
            height=340,
            yaxis_title="Value (INR)",
            xaxis_title="Date",
            margin=dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(
            value_chart,
            use_container_width=True,
        )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Open Positions")

        if open_positions_df.empty:
            st.info("No open paper positions.")
        else:
            display_df = open_positions_df.copy()

            display_df["entry_date"] = pd.to_datetime(
                display_df["entry_date"]
            ).dt.strftime("%d %b %Y")

            for column in [
                "entry_price",
                "initial_stop_price",
                "trailing_stop_price",
                "highest_price_since_entry",
                "current_price",
                "market_value",
                "unrealized_pnl_inr",
            ]:
                display_df[column] = display_df[column].map(format_inr)

            display_df["unrealized_pnl_pct"] = display_df[
                "unrealized_pnl_pct"
            ].map(format_pct)

            display_df = display_df.rename(
                columns={
                    "symbol": "Symbol",
                    "sector": "Sector",
                    "entry_date": "Entry Date",
                    "entry_price": "Entry",
                    "quantity": "Qty",
                    "initial_stop_price": "Initial Stop",
                    "trailing_stop_price": "Trailing Stop",
                    "highest_price_since_entry": "Highest",
                    "current_price": "Current",
                    "market_value": "Market Value",
                    "unrealized_pnl_inr": "Unrealized P&L",
                    "unrealized_pnl_pct": "P&L %",
                }
            )

            st.dataframe(
                display_df[
                    [
                        "Symbol",
                        "Sector",
                        "Entry Date",
                        "Entry",
                        "Qty",
                        "Initial Stop",
                        "Trailing Stop",
                        "Current",
                        "Market Value",
                        "Unrealized P&L",
                        "P&L %",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=360,
            )

    with right_col:
        st.subheader("Portfolio Exposure")

        exposure_pct = float(
            snapshot["gross_exposure_pct"] or 0
        ) * 100

        st.progress(
            min(max(exposure_pct / 100, 0.0), 1.0),
            text=f"Gross Equity Exposure: {exposure_pct:.1f}%",
        )

        st.metric(
            "Total P&L",
            format_inr(snapshot["total_pnl_inr"]),
        )

        st.caption(
            "Exposure and P&L are paper-trading marks based on "
            "latest stored daily closes."
        )

    st.divider()
    st.subheader("Recent Closed Positions")

    if closed_positions_df.empty:
        st.info("No closed paper positions yet.")
    else:
        closed_df = closed_positions_df.copy()

        for date_col in ["entry_date", "exit_date"]:
            closed_df[date_col] = pd.to_datetime(
                closed_df[date_col]
            ).dt.strftime("%d %b %Y")

        for amount_col in [
            "entry_price",
            "exit_price",
            "realized_pnl_inr",
        ]:
            closed_df[amount_col] = closed_df[amount_col].map(
                format_inr
            )

        closed_df["realized_pnl_pct"] = closed_df[
            "realized_pnl_pct"
        ].map(format_pct)

        closed_df = closed_df.rename(
            columns={
                "symbol": "Symbol",
                "sector": "Sector",
                "entry_date": "Entry Date",
                "entry_price": "Entry",
                "quantity": "Qty",
                "exit_date": "Exit Date",
                "exit_price": "Exit",
                "realized_pnl_inr": "Realized P&L",
                "realized_pnl_pct": "P&L %",
                "exit_reason": "Exit Reason",
            }
        )

        st.dataframe(
            closed_df,
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    """Run the Streamlit dashboard."""
    st.title("📈 Indian Stock Trader Dashboard")
    st.caption(
        "Local PostgreSQL-backed daily, weekly, and monthly "
        "NSE equity prediction monitor"
    )

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        try:
            symbols = load_symbols()
        except Exception as exc:
            st.error("Unable to load symbols from PostgreSQL.")
            st.exception(exc)
            st.stop()

        if not symbols:
            st.warning(
                "No symbols found in market_ohlcv. "
                "Run your data-ingestion job first."
            )
            st.stop()

        selected_symbol = st.selectbox(
            "Select NSE symbol",
            options=symbols,
            index=0,
        )

        chart_days = st.selectbox(
            "Chart history",
            options=[60, 90, 180, 365],
            index=2,
            format_func=lambda value: f"Last {value} trading days",
        )

        if st.button("🔄 Refresh dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        selected_page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Market Scanner",
                "Symbol Explorer",
                "Paper Portfolio",
            ],
            index=0,
        )

        st.divider()

        st.caption("Signal legend")
        st.success("BUY = Positive return exceeds model threshold")
        st.warning("HOLD = No high-conviction action")
        st.error("SELL / EXIT = Avoid or exit a delivery holding")

        st.divider()

        st.caption(
            "This dashboard is for research and paper-trading workflows. "
            "It is not financial advice."
        )

    # -------------------------------------------------------------
    # Page routing
    # -------------------------------------------------------------
    if selected_page == "Paper Portfolio":
        render_paper_portfolio()
        return


        st.divider()

        dashboard_page = st.radio(
            "Navigate",
            options=[
                "Overview",
                "Market Scanner",
                "Symbol Explorer",
                "Paper Portfolio",
            ],
            index=0,
        )

        if dashboard_page == "Overview":
            render_overview()

        elif dashboard_page == "Market Scanner":
            render_market_scanner()

        elif dashboard_page == "Symbol Explorer":
            render_symbol_explorer(selected_symbol)

        elif dashboard_page == "Paper Portfolio":
            render_paper_portfolio()

    # Load current selected-symbol data
    try:
        price_df = load_price_data(
            symbol=selected_symbol,
            days=max(365, chart_days + 60),
        )

        predictions_df = load_latest_predictions(
            symbol=selected_symbol,
        )

        history_df = load_prediction_history(
            symbol=selected_symbol,
        )

        entry_decision_df = load_latest_entry_decision(
            symbol=selected_symbol,
        )

    except Exception as exc:
        import traceback

        st.error("Could not load PostgreSQL data.")
        st.exception(exc)

        st.code(
            traceback.format_exc(),
            language="text",
        )
        st.stop()

    if price_df.empty:
        st.warning(
            f"No OHLCV price data is available for {selected_symbol}."
        )
        st.stop()

    latest_price_row = price_df.iloc[-1]
    latest_close = float(latest_price_row["close"])

    previous_close = (
        float(price_df.iloc[-2]["close"])
        if len(price_df) > 1
        else None
    )

    daily_change_pct = (
        ((latest_close / previous_close) - 1) * 100
        if previous_close not in [None, 0]
        else None
    )

    # Header metrics
    left, middle, right, far_right = st.columns(4)

    left.metric(
        "Latest close",
        format_inr(latest_close),
        format_pct(daily_change_pct),
    )

    middle.metric(
        "Last market date",
        latest_price_row["date"].strftime("%d %b %Y"),
    )

    right.metric(
        "Stored price rows",
        f"{len(price_df):,}",
    )

    far_right.metric(
        "Prediction records",
        f"{len(history_df):,}",
    )


    # -------------------------------------------------------------
    # Market-wide Top 10 BUY rankings
    # -------------------------------------------------------------
    st.divider()
    st.header("📊 Raw Model BUY Signals")
    st.caption(
    "These are unfiltered model signals. They do not yet account for "
    "market regime, liquidity, trend, sector limits, existing positions, "
    "or risk-based position sizing. Use the Top 10 Eligible BUY Candidates "
    "section above for paper-trading review."
)

    try:
        latest_prices_df = load_latest_prices()
    except Exception as exc:
        st.error("Could not load latest prices for market-wide rankings.")
        st.exception(exc)
        latest_prices_df = pd.DataFrame(
            columns=["symbol", "date", "close"]
        )

    daily_tab, weekly_tab, monthly_tab = st.tabs([
        "Daily — 1 Trading Day",
        "Weekly — 5 Trading Days",
        "Monthly — 20 Trading Days",
    ])

    with daily_tab:
        show_top_buy_table(
            horizon="daily_1d",
            latest_prices_df=latest_prices_df,
        )

    with weekly_tab:
        show_top_buy_table(
            horizon="weekly_5d",
            latest_prices_df=latest_prices_df,
        )

    with monthly_tab:
        show_top_buy_table(
            horizon="monthly_20d",
            latest_prices_df=latest_prices_df,
        )

    st.divider()

        # -------------------------------------------------------------
    # Final filtered paper-trading shortlist
    # -------------------------------------------------------------
    st.divider()

    try:
        show_top_eligible_buys()
        st.divider()
        show_wait_daily_timing_table()
    except Exception as exc:
        st.error(
            "Could not load final entry decisions. "
            "Run entry_decision_engine.py first."
        )
        st.exception(exc)

    # Prediction cards
    st.subheader(f"{selected_symbol} — Current Model View")

    # -------------------------------------------------------------
    # Final entry decision for the selected stock
    # -------------------------------------------------------------
    if entry_decision_df.empty:
        st.info(
            f"⚪ No entry decision exists for {selected_symbol}. "
            "Run `python predict_weekly_monthly.py` and then "
            "`python entry_decision_engine.py`."
        )

    else:
        decision = entry_decision_df.iloc[0]

        status = str(
            decision.get(
                "entry_status",
                "INSUFFICIENT_DATA",
            )
        )

        rejection_reasons = str(
            decision.get(
                "rejection_reasons",
                "",
            )
        )

        display_status_message(
            status=status,
            symbol=selected_symbol,
            rejection_reasons=rejection_reasons,
        )

        # Final decision / execution-oriented fields
        top_left, top_middle, top_right, top_far_right = st.columns(4)

        top_left.metric(
            "Final Status",
            decision_status_display(status)[0],
        )

        top_middle.metric(
            "Suggested Qty",
            f"{int(decision.get('suggested_quantity', 0)):,}",
        )

        top_right.metric(
            "Signal Reference Close",
            format_inr(decision.get("latest_close")),
        )

        top_far_right.metric(
            "Initial ATR Stop",
            format_inr(decision.get("initial_stop_price")),
        )

        st.caption(
            f"Decision date: "
            f"{pd.to_datetime(decision['decision_date']).strftime('%d %b %Y')} "
            f" | Market regime: **{decision.get('market_regime', 'UNKNOWN')}**"
            f" | Sector: **{decision.get('sector', 'UNKNOWN')}**"
        )

        # Multi-horizon prediction summary
        st.markdown("#### Multi-Horizon Forecast Alignment")

        daily_col, weekly_col, monthly_col = st.columns(3)

        with daily_col:
            daily_return = decision.get(
                "daily_predicted_return"
            )

            daily_signal = decision.get(
                "raw_daily_signal",
                0,
            )

            st.metric(
                "Daily Forecast",
                format_pct(
                    float(daily_return) * 100
                    if pd.notna(daily_return)
                    else None
                ),
            )

            st.caption(
                f"Signal: **{raw_signal_label(daily_signal)}**"
            )

        with weekly_col:
            weekly_return = decision.get(
                "weekly_predicted_return"
            )

            weekly_signal = decision.get(
                "raw_weekly_signal",
                0,
            )

            st.metric(
                "Weekly Forecast",
                format_pct(
                    float(weekly_return) * 100
                    if pd.notna(weekly_return)
                    else None
                ),
            )

            st.caption(
                f"Signal: **{raw_signal_label(weekly_signal)}**"
            )

        with monthly_col:
            monthly_return = decision.get(
                "monthly_predicted_return"
            )

            monthly_signal = decision.get(
                "raw_monthly_signal",
                0,
            )

            st.metric(
                "Monthly Forecast",
                format_pct(
                    float(monthly_return) * 100
                    if pd.notna(monthly_return)
                    else None
                ),
            )

            st.caption(
                f"Signal: **{raw_signal_label(monthly_signal)}**"
            )

        # Detailed technical / risk context
        with st.expander("Entry decision details"):
            detail_left, detail_right = st.columns(2)

            with detail_left:
                st.write(
                    f"**Close:** {format_inr(decision.get('latest_close'))}"
                )

                st.write(
                    f"**SMA 50:** {format_inr(decision.get('sma_50'))}"
                )

                st.write(
                    f"**ATR 14:** {format_inr(decision.get('atr_14'))}"
                )

                st.write(
                    "**20D Avg Traded Value:** "
                    f"{format_inr(decision.get('average_traded_value_inr'))}"
                )

            with detail_right:
                st.write(
                    f"**Suggested Entry:** "
                    f"{format_inr(decision.get('entry_price'))}"
                )

                st.write(
                    f"**Initial Stop:** "
                    f"{format_inr(decision.get('initial_stop_price'))}"
                )

                st.write(
                    f"**Risk per Share:** "
                    f"{format_inr(decision.get('risk_per_share'))}"
                )

                st.write(
                    f"**Suggested Position Value:** "
                    f"{format_inr(decision.get('suggested_position_value'))}"
                )

            st.write(
                f"**Market Index:** "
                f"{decision.get('market_symbol', 'UNKNOWN')}"
            )

            st.write(
                f"**Market Regime:** "
                f"{decision.get('market_regime', 'UNKNOWN')}"
            )

            st.write(
                f"**Model:** "
                f"{decision.get('model_name', 'unknown')} "
                f"({decision.get('model_version', 'v1')})"
            )
    daily_col, weekly_col, monthly_col = st.columns(3)

    with daily_col:
        show_prediction_card(
            horizon="daily_1d",
            row=get_prediction_row(predictions_df, "daily_1d"),
            latest_close=latest_close,
        )

    with weekly_col:
        show_prediction_card(
            horizon="weekly_5d",
            row=get_prediction_row(predictions_df, "weekly_5d"),
            latest_close=latest_close,
        )

    with monthly_col:
        show_prediction_card(
            horizon="monthly_20d",
            row=get_prediction_row(predictions_df, "monthly_20d"),
            latest_close=latest_close,
        )

    # Forecast chart
    st.divider()

    chart_col, signal_col = st.columns([2, 1])

    with chart_col:
        st.plotly_chart(
            create_price_chart(
                price_df=price_df,
                symbol=selected_symbol,
                chart_days=chart_days,
            ),
            use_container_width=True,
        )

    with signal_col:
        return_chart = create_return_comparison_chart(predictions_df)

        if return_chart is not None:
            st.plotly_chart(
                return_chart,
                use_container_width=True,
            )
        else:
            st.info("No prediction records are available yet.")

        st.markdown("### Latest stored predictions")

        if predictions_df.empty:
            st.info("Run your prediction job to populate model_signals.")
        else:
            signal_table = predictions_df[
                [
                    "prediction_horizon",
                    "signal_date",
                    "predicted_return_pct",
                    "signal_label",
                    "model_name",
                ]
            ].copy()

            signal_table = signal_table.rename(
                columns={
                    "prediction_horizon": "Horizon",
                    "signal_date": "Prediction date",
                    "predicted_return_pct": "Return %",
                    "signal_label": "Signal",
                    "model_name": "Model",
                }
            )

            signal_table["Return %"] = signal_table[
                "Return %"
            ].map(format_pct)

            signal_table["Prediction date"] = pd.to_datetime(
                signal_table["Prediction date"]
            ).dt.strftime("%d %b %Y")

            st.dataframe(
                signal_table,
                use_container_width=True,
                hide_index=True,
            )

    # Price history / data table
    st.divider()

    st.subheader("Recent Price Data")

    display_prices = price_df.tail(30).copy()

    display_prices["date"] = display_prices["date"].dt.strftime(
        "%d %b %Y"
    )

    for column in ["open", "high", "low", "close"]:
        display_prices[column] = display_prices[column].map(
            format_inr
        )

    display_prices["volume"] = display_prices["volume"].map(
        lambda value: f"{value:,.0f}"
        if pd.notna(value)
        else "—"
    )

    display_prices = display_prices.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    st.dataframe(
        display_prices.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    # History
    with st.expander("Prediction history and audit trail"):
        if history_df.empty:
            st.info(
                "No prediction history is available for this symbol."
            )
        else:
            display_history = history_df.copy()

            display_history["signal_date"] = display_history[
                "signal_date"
            ].dt.strftime("%d %b %Y")

            display_history["created_at"] = display_history[
                "created_at"
            ].dt.strftime("%d %b %Y %H:%M")

            display_history["predicted_return_pct"] = display_history[
                "predicted_return_pct"
            ].map(format_pct)

            display_history = display_history[
                [
                    "signal_date",
                    "prediction_horizon",
                    "predicted_return_pct",
                    "signal_label",
                    "model_name",
                    "model_version",
                    "created_at",
                ]
            ].rename(
                columns={
                    "signal_date": "Prediction Date",
                    "prediction_horizon": "Horizon",
                    "predicted_return_pct": "Predicted Return",
                    "signal_label": "Signal",
                    "model_name": "Model",
                    "model_version": "Version",
                    "created_at": "Stored At",
                }
            )

            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.caption(
        "Important: Predictions are model estimates. Validate weekly "
        "signals after five future NSE sessions and monthly signals "
        "after twenty sessions. Do not treat dashboard labels as "
        "investment advice or automated trade instructions."
    )


if __name__ == "__main__":
    main()