"""
PostgreSQL repository layer for the Indian Stock Trading System.

Stores:
- Raw daily OHLCV market data
- Engineered features
- Model predictions / trade signals
- Backtest portfolio values
"""

import logging
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from settings import config

logger = logging.getLogger(__name__)


class PostgresDatabase:
    """PostgreSQL data-access layer."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or config.DATABASE_URL

        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

    def test_connection(self) -> bool:
        """Return True only when PostgreSQL is reachable."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection successful.")
            return True
        except Exception as exc:
            logger.exception("PostgreSQL connection failed: %s", exc)
            return False

    def create_schema(self) -> None:
        """Create required tables and indexes if they do not already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id BIGSERIAL PRIMARY KEY,

            portfolio_name VARCHAR(100) NOT NULL DEFAULT 'paper_default',
            symbol VARCHAR(40) NOT NULL,
            sector VARCHAR(150),
            status VARCHAR(20) NOT NULL DEFAULT 'OPEN',

            entry_decision_id BIGINT,
            entry_date TIMESTAMP NOT NULL,
            entry_price DOUBLE PRECISION NOT NULL,
            quantity INTEGER NOT NULL,

            initial_stop_price DOUBLE PRECISION,
            trailing_stop_price DOUBLE PRECISION,
            highest_price_since_entry DOUBLE PRECISION,

            current_price DOUBLE PRECISION,
            market_value DOUBLE PRECISION,
            unrealized_pnl_inr DOUBLE PRECISION,
            unrealized_pnl_pct DOUBLE PRECISION,

            exit_date TIMESTAMP,
            exit_price DOUBLE PRECISION,
            realized_pnl_inr DOUBLE PRECISION,
            realized_pnl_pct DOUBLE PRECISION,
            exit_reason VARCHAR(100),

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT chk_position_status
                CHECK (status IN ('OPEN', 'CLOSED')),

            CONSTRAINT chk_position_quantity
                CHECK (quantity > 0)
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_positions_open
        ON portfolio_positions (portfolio_name, status, symbol);

        CREATE INDEX IF NOT EXISTS idx_portfolio_positions_entry_date
        ON portfolio_positions (entry_date DESC);


        CREATE TABLE IF NOT EXISTS portfolio_transactions (
            id BIGSERIAL PRIMARY KEY,

            portfolio_position_id BIGINT REFERENCES portfolio_positions(id),
            portfolio_name VARCHAR(100) NOT NULL DEFAULT 'paper_default',

            transaction_date TIMESTAMP NOT NULL,
            symbol VARCHAR(40) NOT NULL,
            transaction_type VARCHAR(20) NOT NULL,

            quantity INTEGER NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            gross_value DOUBLE PRECISION NOT NULL,

            estimated_cost_inr DOUBLE PRECISION NOT NULL DEFAULT 0,
            net_value DOUBLE PRECISION NOT NULL,

            reason VARCHAR(150),
            source VARCHAR(100) NOT NULL DEFAULT 'paper_portfolio_engine',

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT chk_transaction_type
                CHECK (transaction_type IN ('BUY', 'SELL'))
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_transactions_symbol_date
        ON portfolio_transactions (symbol, transaction_date DESC);


        CREATE TABLE IF NOT EXISTS portfolio_daily_snapshots (
            id BIGSERIAL PRIMARY KEY,

            portfolio_name VARCHAR(100) NOT NULL DEFAULT 'paper_default',
            snapshot_date DATE NOT NULL,

            cash_inr DOUBLE PRECISION NOT NULL,
            holdings_value_inr DOUBLE PRECISION NOT NULL,
            total_value_inr DOUBLE PRECISION NOT NULL,

            open_positions INTEGER NOT NULL DEFAULT 0,
            gross_exposure_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
            daily_pnl_inr DOUBLE PRECISION,
            total_pnl_inr DOUBLE PRECISION,
            drawdown_pct DOUBLE PRECISION,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_portfolio_snapshot
                UNIQUE (portfolio_name, snapshot_date)
        );

        CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date
        ON portfolio_daily_snapshots (portfolio_name, snapshot_date DESC);

        CREATE TABLE IF NOT EXISTS entry_decisions (
            id BIGSERIAL PRIMARY KEY,

            symbol VARCHAR(40) NOT NULL,
            decision_date TIMESTAMP NOT NULL,

            daily_predicted_return DOUBLE PRECISION,
            weekly_predicted_return DOUBLE PRECISION,
            monthly_predicted_return DOUBLE PRECISION,

            latest_close DOUBLE PRECISION,
            sma_50 DOUBLE PRECISION,
            atr_14 DOUBLE PRECISION,
            average_traded_value_inr DOUBLE PRECISION,

            market_symbol VARCHAR(40),
            market_close DOUBLE PRECISION,
            market_sma_50 DOUBLE PRECISION,
            market_sma_200 DOUBLE PRECISION,
            market_regime VARCHAR(30),

            raw_daily_signal SMALLINT,
            raw_weekly_signal SMALLINT,
            raw_monthly_signal SMALLINT,

            entry_status VARCHAR(30) NOT NULL,
            rejection_reasons TEXT,

            entry_price DOUBLE PRECISION,
            initial_stop_price DOUBLE PRECISION,
            risk_per_share DOUBLE PRECISION,
            suggested_quantity INTEGER,
            suggested_position_value DOUBLE PRECISION,

            current_portfolio_value DOUBLE PRECISION,
            current_portfolio_exposure_pct DOUBLE PRECISION,
            sector VARCHAR(150),
            sector_exposure_pct DOUBLE PRECISION,

            model_name VARCHAR(100),
            model_version VARCHAR(100),

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_entry_decision UNIQUE (
                symbol,
                decision_date,
                model_name,
                model_version
            )
        );

        CREATE INDEX IF NOT EXISTS idx_entry_decisions_date_status
        ON entry_decisions (decision_date DESC, entry_status);

        CREATE INDEX IF NOT EXISTS idx_entry_decisions_symbol_date
        ON entry_decisions (symbol, decision_date DESC);

        CREATE TABLE IF NOT EXISTS market_ohlcv (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(40) NOT NULL,
            trade_date TIMESTAMP NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION,
            dividends DOUBLE PRECISION DEFAULT 0,
            stock_splits DOUBLE PRECISION DEFAULT 0,
            data_source VARCHAR(50) NOT NULL DEFAULT 'yfinance',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_market_ohlcv_symbol_date UNIQUE (symbol, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_market_ohlcv_symbol_date
        ON market_ohlcv (symbol, trade_date DESC);

        CREATE TABLE IF NOT EXISTS model_signals (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(40) NOT NULL,
            signal_date TIMESTAMP NOT NULL,
            prediction_horizon VARCHAR(20) NOT NULL,
            predicted_return DOUBLE PRECISION,
            signal SMALLINT NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(100),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_model_signal UNIQUE (
                symbol,
                signal_date,
                prediction_horizon,
                model_name,
                model_version
            )
        );

        CREATE INDEX IF NOT EXISTS idx_model_signals_symbol_date
        ON model_signals (symbol, signal_date DESC);

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id BIGSERIAL PRIMARY KEY,
            run_name VARCHAR(150) NOT NULL,
            symbol VARCHAR(40),
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            initial_capital DOUBLE PRECISION NOT NULL,
            final_value DOUBLE PRECISION,
            total_return_pct DOUBLE PRECISION,
            sharpe_ratio DOUBLE PRECISION,
            max_drawdown_pct DOUBLE PRECISION,
            total_trades INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """

        with self.engine.begin() as connection:
            for statement in ddl.split(";"):
                statement = statement.strip()
                if statement:
                    connection.execute(text(statement))

        logger.info("PostgreSQL schema is ready.")

    def upsert_market_data(self, df: pd.DataFrame) -> int:
        """
        Insert or update daily OHLCV rows.

        Requires:
        date, symbol, open, high, low, close, volume.

        The unique symbol + trade_date key prevents duplicates.
        """
        if df.empty:
            logger.warning("No market data received for database write.")
            return 0

        required = ["date", "symbol", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(f"Cannot save market data. Missing columns: {missing}")

        data = df.copy()

        data = data.rename(columns={"date": "trade_date"})
        data["trade_date"] = pd.to_datetime(
            data["trade_date"],
            errors="coerce",
        )

        # Remove timezone to avoid inconsistent timestamp representations.
        if getattr(data["trade_date"].dt, "tz", None) is not None:
            data["trade_date"] = data["trade_date"].dt.tz_localize(None)

        for column in ["open", "high", "low", "close", "volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if "dividends" not in data.columns:
            data["dividends"] = 0.0

        if "stock splits" in data.columns:
            data = data.rename(columns={"stock splits": "stock_splits"})

        if "stock_splits" not in data.columns:
            data["stock_splits"] = 0.0

        data["data_source"] = "yfinance"

        columns = [
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividends",
            "stock_splits",
            "data_source",
        ]

        data = data[columns].dropna(
            subset=["symbol", "trade_date", "open", "high", "low", "close"]
        )

        sql = text("""
            INSERT INTO market_ohlcv (
                symbol,
                trade_date,
                open,
                high,
                low,
                close,
                volume,
                dividends,
                stock_splits,
                data_source
            )
            VALUES (
                :symbol,
                :trade_date,
                :open,
                :high,
                :low,
                :close,
                :volume,
                :dividends,
                :stock_splits,
                :data_source
            )
            ON CONFLICT (symbol, trade_date)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                dividends = EXCLUDED.dividends,
                stock_splits = EXCLUDED.stock_splits,
                data_source = EXCLUDED.data_source,
                updated_at = NOW();
        """)

        records = data.to_dict(orient="records")

        with self.engine.begin() as connection:
            connection.execute(sql, records)

        logger.info("Upserted %s OHLCV records to PostgreSQL.", len(records))
        return len(records)

    def get_market_data(
        self,
        symbols: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read stored OHLCV data from PostgreSQL.

        Returned columns match the rest of the project:
        date, symbol, open, high, low, close, volume, dividends, stock_splits.
        """
        conditions = []
        parameters = {}

        if symbols:
            conditions.append("symbol = ANY(:symbols)")
            parameters["symbols"] = list(symbols)

        if start_date:
            conditions.append("trade_date >= :start_date")
            parameters["start_date"] = start_date

        if end_date:
            conditions.append("trade_date <= :end_date")
            parameters["end_date"] = end_date

        where_clause = (
            f"WHERE {' AND '.join(conditions)}"
            if conditions
            else ""
        )

        sql = text(f"""
            SELECT
                symbol,
                trade_date AS date,
                open,
                high,
                low,
                close,
                volume,
                dividends,
                stock_splits,
                data_source
            FROM market_ohlcv
            {where_clause}
            ORDER BY symbol, trade_date;
        """)

        df = pd.read_sql(sql, self.engine, params=parameters)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        logger.info("Loaded %s OHLCV records from PostgreSQL.", len(df))
        return df

    def save_entry_decisions(
    self,
    decisions_df: pd.DataFrame,
    ) -> int:
        """
        Save entry-decision results to PostgreSQL.

        This method records both:
        - ELIGIBLE_BUY records
        - BLOCKED records and the reasons they were rejected.
        """
        if decisions_df.empty:
            logger.warning("No entry decisions to save.")
            return 0

        data = decisions_df.copy()

        required_columns = [
            "symbol",
            "decision_date",
            "entry_status",
            "model_name",
            "model_version",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Cannot save entry decisions. Missing columns: {missing}"
            )

        optional_columns = [
            "daily_predicted_return",
            "raw_daily_signal",
            "weekly_predicted_return",
            "monthly_predicted_return",
            "latest_close",
            "sma_50",
            "atr_14",
            "average_traded_value_inr",
            "market_symbol",
            "market_close",
            "market_sma_50",
            "market_sma_200",
            "market_regime",
            "raw_weekly_signal",
            "raw_monthly_signal",
            "rejection_reasons",
            "entry_price",
            "initial_stop_price",
            "risk_per_share",
            "suggested_quantity",
            "suggested_position_value",
            "current_portfolio_value",
            "current_portfolio_exposure_pct",
            "sector",
            "sector_exposure_pct",
        ]

        for column in optional_columns:
            if column not in data.columns:
                data[column] = None

        data["decision_date"] = pd.to_datetime(
            data["decision_date"],
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
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        for column in [
            "raw_daily_signal",
            "raw_weekly_signal",
            "raw_monthly_signal",
            "suggested_quantity",
        ]:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            ).fillna(0).astype(int)

        fields = [
            "symbol",
            "decision_date",
            "daily_predicted_return",
            "weekly_predicted_return",
            "monthly_predicted_return",
            "latest_close",
            "sma_50",
            "atr_14",
            "average_traded_value_inr",
            "market_symbol",
            "market_close",
            "market_sma_50",
            "market_sma_200",
            "market_regime",
            "raw_daily_signal",
            "raw_weekly_signal",
            "raw_monthly_signal",
            "entry_status",
            "rejection_reasons",
            "entry_price",
            "initial_stop_price",
            "risk_per_share",
            "suggested_quantity",
            "suggested_position_value",
            "current_portfolio_value",
            "current_portfolio_exposure_pct",
            "sector",
            "sector_exposure_pct",
            "model_name",
            "model_version",
        ]

        sql = text("""
            INSERT INTO entry_decisions (
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
                model_version
            )
            VALUES (
                :symbol,
                :decision_date,
                :daily_predicted_return,
                :weekly_predicted_return,
                :monthly_predicted_return,
                :latest_close,
                :sma_50,
                :atr_14,
                :average_traded_value_inr,
                :market_symbol,
                :market_close,
                :market_sma_50,
                :market_sma_200,
                :market_regime,
                :raw_daily_signal,
                :raw_weekly_signal,
                :raw_monthly_signal,
                :entry_status,
                :rejection_reasons,
                :entry_price,
                :initial_stop_price,
                :risk_per_share,
                :suggested_quantity,
                :suggested_position_value,
                :current_portfolio_value,
                :current_portfolio_exposure_pct,
                :sector,
                :sector_exposure_pct,
                :model_name,
                :model_version
            )
            ON CONFLICT (
                symbol,
                decision_date,
                model_name,
                model_version
            )
            DO UPDATE SET
                daily_predicted_return = EXCLUDED.daily_predicted_return,
                weekly_predicted_return = EXCLUDED.weekly_predicted_return,
                monthly_predicted_return = EXCLUDED.monthly_predicted_return,
                latest_close = EXCLUDED.latest_close,
                sma_50 = EXCLUDED.sma_50,
                atr_14 = EXCLUDED.atr_14,
                average_traded_value_inr = EXCLUDED.average_traded_value_inr,
                market_symbol = EXCLUDED.market_symbol,
                market_close = EXCLUDED.market_close,
                market_sma_50 = EXCLUDED.market_sma_50,
                market_sma_200 = EXCLUDED.market_sma_200,
                market_regime = EXCLUDED.market_regime,
                raw_daily_signal = EXCLUDED.raw_daily_signal,
                raw_weekly_signal = EXCLUDED.raw_weekly_signal,
                raw_monthly_signal = EXCLUDED.raw_monthly_signal,
                entry_status = EXCLUDED.entry_status,
                rejection_reasons = EXCLUDED.rejection_reasons,
                entry_price = EXCLUDED.entry_price,
                initial_stop_price = EXCLUDED.initial_stop_price,
                risk_per_share = EXCLUDED.risk_per_share,
                suggested_quantity = EXCLUDED.suggested_quantity,
                suggested_position_value = EXCLUDED.suggested_position_value,
                current_portfolio_value = EXCLUDED.current_portfolio_value,
                current_portfolio_exposure_pct = EXCLUDED.current_portfolio_exposure_pct,
                sector = EXCLUDED.sector,
                sector_exposure_pct = EXCLUDED.sector_exposure_pct,
                created_at = NOW();
        """)

        data = data[fields].dropna(
            subset=[
                "symbol",
                "decision_date",
                "entry_status",
                "model_name",
                "model_version",
            ]
        )

        with self.engine.begin() as connection:
            connection.execute(
                sql,
                data.to_dict(orient="records"),
            )

        logger.info("Saved %s entry decisions.", len(data))
        return len(data)
    
    def save_signals(
        self,
        signals_df: pd.DataFrame,
        horizon: str = "5d",
        model_name: str = "lstm_xgboost_ensemble",
        model_version: str = "v1",
    ) -> int:
        """
        Store generated buy/sell/hold signals.

        Required input columns:
        date, symbol, prediction, signal.
        """
        if signals_df.empty:
            return 0

        required = ["date", "symbol", "prediction", "signal"]
        missing = [column for column in required if column not in signals_df.columns]

        if missing:
            raise ValueError(f"Cannot save signals. Missing columns: {missing}")

        data = signals_df.copy().rename(
            columns={
                "date": "signal_date",
                "prediction": "predicted_return",
            }
        )

        data["signal_date"] = pd.to_datetime(data["signal_date"])
        data["prediction_horizon"] = horizon
        data["model_name"] = model_name
        data["model_version"] = model_version

        sql = text("""
            INSERT INTO model_signals (
                symbol,
                signal_date,
                prediction_horizon,
                predicted_return,
                signal,
                model_name,
                model_version
            )
            VALUES (
                :symbol,
                :signal_date,
                :prediction_horizon,
                :predicted_return,
                :signal,
                :model_name,
                :model_version
            )
            ON CONFLICT (
                symbol,
                signal_date,
                prediction_horizon,
                model_name,
                model_version
            )
            DO UPDATE SET
                predicted_return = EXCLUDED.predicted_return,
                signal = EXCLUDED.signal,
                created_at = NOW();
        """)

        fields = [
            "symbol",
            "signal_date",
            "prediction_horizon",
            "predicted_return",
            "signal",
            "model_name",
            "model_version",
        ]

        with self.engine.begin() as connection:
            connection.execute(
                sql,
                data[fields].to_dict(orient="records"),
            )

        return len(data)