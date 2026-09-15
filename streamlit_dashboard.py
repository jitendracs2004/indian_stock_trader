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
    Load the latest final entry decision for one symbol.

    Reads from entry_decisions, which is created by:
        python entry_decision_engine.py
    """
    db = get_database()

    query = text("""
        SELECT
            symbol,
            decision_date,
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
            created_at DESC
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

    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "suggested_quantity" in df.columns:
        df["suggested_quantity"] = pd.to_numeric(
            df["suggested_quantity"],
            errors="coerce",
        ).fillna(0).astype(int)

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


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
def format_inr(value: float | None) -> str:
    """Format a numeric value in INR."""
    if value is None or pd.isna(value):
        return "—"

    return f"₹{value:,.2f}"


def format_pct(value: float | None) -> str:
    """Format percentage with plus/minus sign."""
    if value is None or pd.isna(value):
        return "—"

    return f"{value:+.2f}%"


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

        st.caption("Signal legend")
        st.success("BUY = Positive return exceeds model threshold")
        st.warning("HOLD = No high-conviction action")
        st.error("SELL / EXIT = Avoid or exit a delivery holding")

        st.divider()

        st.caption(
            "This dashboard is for research and paper-trading workflows. "
            "It is not financial advice."
        )

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
    except Exception as exc:
        st.error(
            "Could not load final entry decisions. "
            "Run entry_decision_engine.py first."
        )
        st.exception(exc)

    # Prediction cards
    st.subheader(f"{selected_symbol} — Current Model View")

        # -------------------------------------------------------------
    # Final entry decision for selected stock
    # -------------------------------------------------------------
    if entry_decision_df.empty:
        st.info(
            "No final entry decision exists for this symbol. "
            "Run `python entry_decision_engine.py` after generating "
            "weekly and monthly predictions."
        )
    else:
        decision = entry_decision_df.iloc[0]

        if decision["entry_status"] == "ELIGIBLE_BUY":
            st.success(
                f"✅ Final decision: ELIGIBLE BUY — "
                f"{selected_symbol} passed all nine entry checks."
            )
        else:
            st.warning(
                f"⛔ Final decision: BLOCKED — "
                f"{selected_symbol} is not eligible for a new entry."
            )

        decision_left, decision_middle, decision_right, decision_far_right = st.columns(4)

        decision_left.metric(
            "Final status",
            decision["entry_status"],
        )

        decision_middle.metric(
            "Suggested quantity",
            f"{int(decision['suggested_quantity']):,}",
        )

        decision_right.metric(
            "Suggested entry",
            format_inr(decision["entry_price"]),
        )

        decision_far_right.metric(
            "Initial stop",
            format_inr(decision["initial_stop_price"]),
        )

        st.caption(
            f"Market regime: **{decision['market_regime']}** | "
            f"Sector: **{decision['sector']}** | "
            f"Decision date: "
            f"{pd.to_datetime(decision['decision_date']).strftime('%d %b %Y')}"
        )

        if decision["entry_status"] != "ELIGIBLE_BUY":
            rejection_text = decision.get(
                "rejection_reasons",
                "No rejection reason stored.",
            )

            st.error(
                f"Entry blocked because: `{rejection_text}`"
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