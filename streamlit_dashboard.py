"""
Streamlit dashboard for Indian Stock Trader.

Reads local PostgreSQL data:
- market_ohlcv: historical daily NSE/BSE price data
- model_signals: daily, weekly, monthly model predictions
- entry_decisions: final 9-condition entry engine output

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
# Theme / styling
# ---------------------------------------------------------------------
def inject_theme() -> None:
    """Inject a compact professional stylesheet."""
    st.markdown(
        """
        <style>
        /* Base typography */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system,
                         sans-serif;
        }

        /* Tighten the main container width and padding */
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* Section headings */
        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.01em;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }

        /* Bordered containers behave like cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
            border-color: #e2e8f0 !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.95rem;
        }

        /* Dataframe corners */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        /* Custom header bar */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 24px;
            border-radius: 14px;
            background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 100%);
            color: #ffffff;
            margin-bottom: 8px;
        }
        .app-header .title {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .app-header .subtitle {
            font-size: 0.9rem;
            opacity: 0.85;
            margin-top: 2px;
        }
        .app-header .badge {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Signal pill */
        .signal-pill {
            display: inline-block;
            color: #ffffff;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }

        /* Section label above headers */
        .section-eyebrow {
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(symbol_count: int) -> None:
    """Render the branded top header bar."""
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <p class="title">Indian Stock Trader</p>
                <p class="subtitle">
                    NSE equity prediction &amp; entry-decision monitor
                </p>
            </div>
            <div class="badge">{symbol_count:,} symbols tracked</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(eyebrow: str, title: str, caption: str = "") -> None:
    """Render a consistent section header with an eyebrow label."""
    st.markdown(
        f'<p class="section-eyebrow">{eyebrow}</p>',
        unsafe_allow_html=True,
    )
    st.subheader(title)
    if caption:
        st.caption(caption)


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
def load_latest_entry_decision(symbol: str) -> pd.DataFrame:
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
def load_top_eligible_buys(limit_rows: int = 10) -> pd.DataFrame:
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


@st.cache_data(ttl=60)
def load_paper_transactions(limit_rows: int = 100) -> pd.DataFrame:
    """Load the paper portfolio BUY/SELL transaction log."""
    db = get_database()

    query = text("""
        SELECT
            transaction_date,
            symbol,
            transaction_type,
            quantity,
            price,
            gross_value,
            estimated_cost_inr,
            net_value,
            reason,
            source
        FROM portfolio_transactions
        WHERE portfolio_name = 'paper_default'
        ORDER BY transaction_date DESC, created_at DESC
        LIMIT :limit_rows;
    """)

    with db.engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"limit_rows": limit_rows},
        )

    if not df.empty:
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"],
            errors="coerce",
        )
        for column in [
            "price",
            "gross_value",
            "estimated_cost_inr",
            "net_value",
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

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


def signal_pill_html(signal_code: int, label: str) -> str:
    """Build an inline colored pill for a signal label."""
    color = SIGNAL_COLOR_MAP.get(signal_code, "#6b7280")
    return (
        f'<span class="signal-pill" style="background-color:{color};">'
        f"{label}</span>"
    )


def _style_return(value: str) -> str:
    """Color a formatted +/- percentage string green/red."""
    if isinstance(value, str) and value.startswith("+"):
        return "color: #16a34a; font-weight: 600;"
    if isinstance(value, str) and value.startswith("-"):
        return "color: #dc2626; font-weight: 600;"
    return ""


def _style_signal(value: str) -> str:
    """Color a signal label cell."""
    mapping = {
        "BUY": "color: #16a34a; font-weight: 700;",
        "HOLD": "color: #b45309; font-weight: 600;",
        "SELL / EXIT": "color: #dc2626; font-weight: 700;",
    }
    return mapping.get(value, "")


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
        title=f"{symbol} — Daily Price",
        height=520,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (INR)",
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
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
        height=340,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        yaxis_title="Predicted return (%)",
        xaxis_title="Horizon",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return figure


# ---------------------------------------------------------------------
# Component: prediction card
# ---------------------------------------------------------------------
def show_prediction_card(
    horizon: str,
    row: pd.Series | None,
    latest_close: float | None,
) -> None:
    """Render one daily/weekly/monthly prediction card."""
    config = HORIZON_CONFIG[horizon]

    with st.container(border=True):
        st.markdown(f"**{config['label']}**")
        st.caption(config["sub_label"])

        if row is None:
            st.info(
                f"No {config['label'].lower()} prediction stored yet."
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
        )

        st.metric(
            label="Expected price",
            value=format_inr(expected_price),
        )

        st.markdown(
            signal_pill_html(signal_code, signal_label),
            unsafe_allow_html=True,
        )

        st.caption(
            f"As of {row['signal_date'].strftime('%d %b %Y')} · "
            f"{row['model_name']} ({row['model_version']})"
        )


# ---------------------------------------------------------------------
# Component: top raw BUY table
# ---------------------------------------------------------------------
def show_top_buy_table(
    horizon: str,
    latest_prices_df: pd.DataFrame,
) -> None:
    """Display the Top 10 latest BUY recommendations for a horizon."""
    horizon_info = HORIZON_CONFIG[horizon]

    top_buy_df = load_top_buy_predictions(
        horizon=horizon,
        limit_rows=10,
    )

    if top_buy_df.empty:
        st.info(
            f"No active BUY signals currently exist for "
            f"{horizon_info['label'].lower()} predictions."
        )
        return

    prices = latest_prices_df[["symbol", "close"]].copy()
    prices = prices.rename(columns={"close": "latest_close_inr"})

    display_df = top_buy_df.merge(prices, on="symbol", how="left")

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

    styled = display_df.style.map(
        _style_return,
        subset=["Predicted Return"],
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=False,
    )

    st.caption(
        "Ranked by model-predicted return among the latest active BUY "
        "signals. Research ranking, not financial advice."
    )


# ---------------------------------------------------------------------
# Component: top eligible buys (filtered shortlist)
# ---------------------------------------------------------------------
def show_top_eligible_buys() -> None:
    """Show only stocks that passed all nine entry checks."""
    eligible_df = load_top_eligible_buys(limit_rows=10)

    section_title(
        "Paper-trading shortlist",
        "Top 10 Eligible BUY Candidates",
        "Stocks that cleared every entry-engine check: weekly forecast, "
        "cost/error buffer, monthly confirmation, SMA-50 trend, market "
        "regime, liquidity, duplicate-position, exposure limits, and "
        "ATR-based sizing.",
    )

    if eligible_df.empty:
        st.warning(
            "No stocks currently qualify as ELIGIBLE_BUY. This can be "
            "normal when the market is bearish, signals are weak, or "
            "liquidity/trend filters block entries."
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
        by=["weekly_predicted_return", "monthly_predicted_return"],
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
        display_df[column] = display_df[column].map(format_inr)

    for column in [
        "weekly_return_pct",
        "monthly_return_pct",
        "stop_distance_pct",
    ]:
        display_df[column] = display_df[column].map(format_pct)

    display_df["suggested_quantity"] = display_df[
        "suggested_quantity"
    ].map(lambda value: f"{int(value):,}")

    display_df["market_regime"] = display_df[
        "market_regime"
    ].fillna("UNKNOWN")
    display_df["sector"] = display_df["sector"].fillna("UNKNOWN")

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

    styled = display_df.style.map(
        _style_return,
        subset=["Weekly Forecast", "Monthly Forecast"],
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=False,
        height=420,
    )

    st.caption(
        "Suggested Entry is the latest close used by the research "
        "engine, not a guaranteed executable price. Initial Stop is "
        "ATR-based. Suggested quantity is a paper-trading allocation."
    )


# ---------------------------------------------------------------------
# Tab: Market Overview
# ---------------------------------------------------------------------
def render_raw_signals() -> None:
    """Tab 1 — unfiltered market-wide raw model BUY rankings."""
    section_title(
        "Unfiltered model output",
        "Top 10 Raw Model BUY Signals",
        "Straight model signals before market regime, liquidity, trend, "
        "sector limits, positions, or risk sizing are applied. See the "
        "Entry Decisions tab for the filtered paper-trading shortlist.",
    )

    try:
        latest_prices_df = load_latest_prices()
    except Exception as exc:
        st.error("Could not load latest prices for rankings.")
        st.exception(exc)
        latest_prices_df = pd.DataFrame(
            columns=["symbol", "date", "close"]
        )

    daily_tab, weekly_tab, monthly_tab = st.tabs([
        "Daily · 1D",
        "Weekly · 5D",
        "Monthly · 20D",
    ])

    with daily_tab:
        show_top_buy_table("daily_1d", latest_prices_df)
    with weekly_tab:
        show_top_buy_table("weekly_5d", latest_prices_df)
    with monthly_tab:
        show_top_buy_table("monthly_20d", latest_prices_df)


# ---------------------------------------------------------------------
# Tab: Entry Decisions
# ---------------------------------------------------------------------
def render_entry_decisions(selected_symbol: str) -> None:
    """Tab 2 — eligible-buy shortlist plus the selected-symbol decision."""
    # Market-wide filtered shortlist.
    try:
        show_top_eligible_buys()
    except Exception as exc:
        st.error(
            "Could not load final entry decisions. "
            "Run entry_decision_engine.py first."
        )
        st.exception(exc)

    st.divider()

    # Selected-symbol final entry decision.
    section_title(
        "Selected symbol",
        f"Entry Decision — {selected_symbol}",
    )

    try:
        entry_decision_df = load_latest_entry_decision(
            symbol=selected_symbol,
        )
    except Exception as exc:
        st.error("Could not load the entry decision for this symbol.")
        st.exception(exc)
        return

    if entry_decision_df.empty:
        st.info(
            "No final entry decision exists for this symbol. Run "
            "`python entry_decision_engine.py` after generating weekly "
            "and monthly predictions."
        )
        return

    decision = entry_decision_df.iloc[0]
    status = decision["entry_status"]

    if status == "ELIGIBLE_BUY":
        st.success(
            f"Final decision: ELIGIBLE BUY — {selected_symbol} "
            "passed all entry checks."
        )
    elif status == "WAIT_DAILY_TIMING":
        st.info(
            f"Final decision: WAIT — {selected_symbol} clears the "
            "weekly/monthly thesis, but the daily signal suggests "
            "waiting for better immediate timing."
        )
    else:
        st.warning(
            f"Final decision: BLOCKED — {selected_symbol} is not "
            "eligible for a new entry."
        )

    d_left, d_mid, d_right, d_far = st.columns(4)
    d_left.metric("Final status", status)
    d_mid.metric(
        "Suggested quantity",
        f"{int(decision['suggested_quantity']):,}",
    )
    d_right.metric(
        "Suggested entry",
        format_inr(decision["entry_price"]),
    )
    d_far.metric(
        "Initial stop",
        format_inr(decision["initial_stop_price"]),
    )

    # Horizon signal pills (daily is the timing gate)
    st.markdown("**Model signals by horizon**")
    pill_daily, pill_weekly, pill_monthly = st.columns(3)

    def _horizon_pill(column, label, signal_col, return_col):
        raw = decision.get(signal_col)
        ret = decision.get(return_col)
        with column:
            if pd.isna(raw):
                st.markdown(
                    f"{label}: "
                    + signal_pill_html(0, "N/A"),
                    unsafe_allow_html=True,
                )
            else:
                code = int(raw)
                st.markdown(
                    f"{label}: "
                    + signal_pill_html(
                        code, SIGNAL_MAP.get(code, "UNKNOWN")
                    ),
                    unsafe_allow_html=True,
                )
            if ret is not None and not pd.isna(ret):
                st.caption(f"Forecast: {format_pct(float(ret) * 100)}")

    _horizon_pill(
        pill_daily, "Daily",
        "raw_daily_signal", "daily_predicted_return",
    )
    _horizon_pill(
        pill_weekly, "Weekly",
        "raw_weekly_signal", "weekly_predicted_return",
    )
    _horizon_pill(
        pill_monthly, "Monthly",
        "raw_monthly_signal", "monthly_predicted_return",
    )

    st.caption(
        f"Market regime: **{decision['market_regime']}** · "
        f"Sector: **{decision['sector']}** · Decision date: "
        f"{pd.to_datetime(decision['decision_date']).strftime('%d %b %Y')}"
    )

    if status not in ("ELIGIBLE_BUY", "WAIT_DAILY_TIMING"):
        rejection_text = decision.get(
            "rejection_reasons",
            "No rejection reason stored.",
        )
        st.error(f"Entry blocked because: `{rejection_text}`")


# ---------------------------------------------------------------------
# Tab: Stock Analysis
# ---------------------------------------------------------------------
def render_stock_analysis(
    selected_symbol: str,
    chart_days: int,
) -> None:
    """Tab 3 — single-symbol deep dive (price, forecasts, history)."""
    try:
        price_df = load_price_data(
            symbol=selected_symbol,
            days=max(365, chart_days + 60),
        )
        predictions_df = load_latest_predictions(symbol=selected_symbol)
        history_df = load_prediction_history(symbol=selected_symbol)
    except Exception as exc:
        import traceback

        st.error("Could not load PostgreSQL data.")
        st.exception(exc)
        st.code(traceback.format_exc(), language="text")
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
    section_title(
        "Selected symbol",
        f"{selected_symbol}",
    )

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

    # Prediction cards
    st.divider()
    st.markdown("#### Model View by Horizon")

    daily_col, weekly_col, monthly_col = st.columns(3)
    with daily_col:
        show_prediction_card(
            "daily_1d",
            get_prediction_row(predictions_df, "daily_1d"),
            latest_close,
        )
    with weekly_col:
        show_prediction_card(
            "weekly_5d",
            get_prediction_row(predictions_df, "weekly_5d"),
            latest_close,
        )
    with monthly_col:
        show_prediction_card(
            "monthly_20d",
            get_prediction_row(predictions_df, "monthly_20d"),
            latest_close,
        )

    # Chart + forecast comparison
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
            st.plotly_chart(return_chart, use_container_width=True)
        else:
            st.info("No prediction records are available yet.")

        st.markdown("##### Latest stored predictions")
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

            signal_table["Return %"] = signal_table["Return %"].map(
                format_pct
            )
            signal_table["Prediction date"] = pd.to_datetime(
                signal_table["Prediction date"]
            ).dt.strftime("%d %b %Y")

            styled_signals = signal_table.style.map(
                _style_return, subset=["Return %"]
            ).map(_style_signal, subset=["Signal"])

            st.dataframe(
                styled_signals,
                use_container_width=True,
                hide_index=True,
            )

    # Recent price data
    st.divider()
    st.markdown("#### Recent Price Data")

    display_prices = price_df.tail(30).copy()
    display_prices["date"] = display_prices["date"].dt.strftime(
        "%d %b %Y"
    )
    for column in ["open", "high", "low", "close"]:
        display_prices[column] = display_prices[column].map(format_inr)
    display_prices["volume"] = display_prices["volume"].map(
        lambda value: f"{value:,.0f}" if pd.notna(value) else "—"
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

    # History audit trail
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

            styled_history = display_history.style.map(
                _style_return, subset=["Predicted Return"]
            ).map(_style_signal, subset=["Signal"])

            st.dataframe(
                styled_history,
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------
# Tab: Paper Portfolio
# ---------------------------------------------------------------------
def render_paper_portfolio() -> None:
    """Tab 4 — paper portfolio tracking (research only)."""
    section_title(
        "Research only",
        "Paper Portfolio",
        "Portfolio tracking based on stored daily closes. "
        "No live broker orders are sent.",
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

    # Portfolio value trend
    if not history_df.empty:
        value_chart = go.Figure()
        value_chart.add_trace(
            go.Scatter(
                x=history_df["snapshot_date"],
                y=history_df["total_value_inr"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#2563eb", width=3),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
            )
        )
        value_chart.update_layout(
            title="Paper Portfolio Value",
            template="plotly_white",
            height=340,
            font=dict(family="Inter, Segoe UI, sans-serif", size=12),
            yaxis_title="Value (INR)",
            xaxis_title="Date",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(value_chart, use_container_width=True)

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("#### Open Positions")

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

            display_df = display_df[
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
            ]

            styled_open = display_df.style.map(
                _style_return, subset=["P&L %"]
            )

            st.dataframe(
                styled_open,
                use_container_width=True,
                hide_index=True,
                height=360,
            )

    with right_col:
        st.markdown("#### Exposure")

        exposure_pct = float(
            snapshot["gross_exposure_pct"] or 0
        ) * 100

        st.progress(
            min(max(exposure_pct / 100, 0.0), 1.0),
            text=f"Gross equity exposure: {exposure_pct:.1f}%",
        )

        st.metric(
            "Total P&L",
            format_inr(snapshot["total_pnl_inr"]),
        )

        st.caption(
            "Exposure and P&L are paper-trading marks based on the "
            "latest stored daily closes."
        )

    st.divider()
    st.markdown("#### Recent Closed Positions")

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
            closed_df[amount_col] = closed_df[amount_col].map(format_inr)

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

        styled_closed = closed_df.style.map(
            _style_return, subset=["P&L %"]
        )

        st.dataframe(
            styled_closed,
            use_container_width=True,
            hide_index=True,
        )

    # Transaction log
    st.divider()
    st.markdown("#### Transaction Log")

    transactions_df = load_paper_transactions()

    if transactions_df.empty:
        st.info("No paper transactions recorded yet.")
    else:
        txn_df = transactions_df.copy()

        txn_df["transaction_date"] = txn_df[
            "transaction_date"
        ].dt.strftime("%d %b %Y")

        for amount_col in [
            "price",
            "gross_value",
            "estimated_cost_inr",
            "net_value",
        ]:
            txn_df[amount_col] = txn_df[amount_col].map(format_inr)

        txn_df = txn_df.rename(
            columns={
                "transaction_date": "Date",
                "symbol": "Symbol",
                "transaction_type": "Type",
                "quantity": "Qty",
                "price": "Price",
                "gross_value": "Gross Value",
                "estimated_cost_inr": "Est. Cost",
                "net_value": "Net Value",
                "reason": "Reason",
                "source": "Source",
            }
        )

        def _style_txn_type(value: str) -> str:
            if value == "BUY":
                return "color: #16a34a; font-weight: 700;"
            if value == "SELL":
                return "color: #dc2626; font-weight: 700;"
            return ""

        styled_txn = txn_df.style.map(_style_txn_type, subset=["Type"])

        st.dataframe(
            styled_txn,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    """Run the Streamlit dashboard."""
    inject_theme()

    # Sidebar
    with st.sidebar:
        st.markdown("### Controls")

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
            "NSE symbol",
            options=symbols,
            index=0,
        )

        chart_days = st.selectbox(
            "Chart history",
            options=[60, 90, 180, 365],
            index=2,
            format_func=lambda value: f"Last {value} trading days",
        )

        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.markdown("**Signal legend**")
        st.markdown(
            signal_pill_html(1, "BUY")
            + " Positive return above threshold",
            unsafe_allow_html=True,
        )
        st.markdown(
            signal_pill_html(0, "HOLD") + " No high-conviction action",
            unsafe_allow_html=True,
        )
        st.markdown(
            signal_pill_html(-1, "SELL / EXIT")
            + " Avoid or exit a holding",
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(
            "For research and paper-trading workflows. "
            "Not financial advice."
        )

    # Header
    render_header(len(symbols))

    # Top-level tabs
    raw_tab, entry_tab, analysis_tab, portfolio_tab = st.tabs([
        "Raw Signals",
        "Entry Decisions",
        "Stock Analysis",
        "Paper Portfolio",
    ])

    with raw_tab:
        render_raw_signals()

    with entry_tab:
        render_entry_decisions(selected_symbol)

    with analysis_tab:
        render_stock_analysis(selected_symbol, chart_days)

    with portfolio_tab:
        render_paper_portfolio()

    # Global footer
    st.divider()
    st.caption(
        "Important: Predictions are model estimates. Validate weekly "
        "signals after five future NSE sessions and monthly signals "
        "after twenty sessions. Do not treat dashboard labels as "
        "investment advice or automated trade instructions."
    )


if __name__ == "__main__":
    main()
