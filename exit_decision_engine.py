"""
Paper Portfolio Exit Engine.

Checks open paper positions and closes them on:
- Initial/trailing stop
- Weekly model SELL / EXIT
- Monthly model SELL / EXIT
- Bearish-market risk exit for losing positions
- Weekly/monthly time stops

This module does NOT send broker orders.
"""

import logging
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from database import PostgresDatabase


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

PORTFOLIO_NAME = "paper_default"
PAPER_TRADE_COST_RATE = 0.0016

WEEKLY_TIME_STOP_DAYS = 7
MONTHLY_TIME_STOP_DAYS = 25

# Treat less than +0.25% as flat after expected horizon.
MINIMUM_PROGRESS_PCT = 0.25


class ExitDecisionEngine:
    """Evaluate and close paper positions."""

    def __init__(
        self,
        db: PostgresDatabase,
        portfolio_name: str = PORTFOLIO_NAME,
    ):
        self.db = db
        self.portfolio_name = portfolio_name

    def get_open_positions(self) -> pd.DataFrame:
        query = text("""
            SELECT
                id,
                portfolio_name,
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
                unrealized_pnl_pct
            FROM portfolio_positions
            WHERE portfolio_name = :portfolio_name
              AND status = 'OPEN'
            ORDER BY entry_date ASC;
        """)

        with self.db.engine.connect() as connection:
            return pd.read_sql(
                query,
                connection,
                params={"portfolio_name": self.portfolio_name},
            )

    def get_latest_close(self, symbol: str) -> tuple[pd.Timestamp, float] | None:
        query = text("""
            SELECT trade_date, close
            FROM market_ohlcv
            WHERE symbol = :symbol
            ORDER BY trade_date DESC
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            row = connection.execute(
                query,
                {"symbol": symbol},
            ).fetchone()

        if row is None:
            return None

        return pd.Timestamp(row[0]), float(row[1])

    def get_latest_model_signal(
        self,
        symbol: str,
        horizon: str,
    ) -> int:
        query = text("""
            SELECT signal
            FROM model_signals
            WHERE symbol = :symbol
              AND prediction_horizon = :horizon
            ORDER BY signal_date DESC, created_at DESC
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            value = connection.execute(
                query,
                {
                    "symbol": symbol,
                    "horizon": horizon,
                },
            ).scalar()

        return int(value) if value is not None else 0

    def get_market_regime(self) -> str:
        query = text("""
            SELECT market_regime
            FROM entry_decisions
            WHERE market_regime IS NOT NULL
            ORDER BY decision_date DESC, created_at DESC
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            regime = connection.execute(query).scalar()

        return str(regime) if regime else "UNKNOWN"

    def get_position_age(self, entry_date: pd.Timestamp) -> int:
        """Approximate age in calendar days for simple paper-trading rules."""
        return max(
            0,
            (datetime.now().date() - pd.Timestamp(entry_date).date()).days,
        )

    def get_exit_reason(
        self,
        position: pd.Series,
        current_price: float,
        market_regime: str,
    ) -> str | None:
        """Return exit reason or None if position remains open."""
        initial_stop = float(position["initial_stop_price"])
        trailing_stop = float(
            position["trailing_stop_price"] or initial_stop
        )

        stop_price = max(initial_stop, trailing_stop)

        if current_price <= stop_price:
            return "STOP_LOSS_OR_TRAILING_STOP"

        weekly_signal = self.get_latest_model_signal(
            position["symbol"],
            "weekly_5d",
        )

        if weekly_signal == -1:
            return "WEEKLY_MODEL_REVERSAL"

        monthly_signal = self.get_latest_model_signal(
            position["symbol"],
            "monthly_20d",
        )

        if monthly_signal == -1:
            return "MONTHLY_MODEL_REVERSAL"

        entry_price = float(position["entry_price"])

        pnl_pct = (
            (current_price / entry_price) - 1
        ) * 100

        age_days = self.get_position_age(position["entry_date"])

        if (
            age_days >= WEEKLY_TIME_STOP_DAYS
            and pnl_pct <= MINIMUM_PROGRESS_PCT
            and weekly_signal != 1
        ):
            return "WEEKLY_TIME_STOP"

        if (
            age_days >= MONTHLY_TIME_STOP_DAYS
            and pnl_pct <= MINIMUM_PROGRESS_PCT
            and monthly_signal != 1
        ):
            return "MONTHLY_TIME_STOP"

        if market_regime == "BEARISH" and pnl_pct < 0:
            return "BEARISH_MARKET_LOSING_POSITION"

        return None

    def close_position(
        self,
        position: pd.Series,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Record paper sell transaction and close the position."""
        quantity = int(position["quantity"])

        # Paper sell price includes one-side cost/slippage approximation.
        effective_exit_price = exit_price * (
            1 - PAPER_TRADE_COST_RATE / 2
        )

        gross_value = effective_exit_price * quantity
        estimated_cost = gross_value * (
            PAPER_TRADE_COST_RATE / 2
        )

        net_value = gross_value - estimated_cost

        entry_cost_basis = (
            float(position["entry_price"])
            * quantity
            * (1 + PAPER_TRADE_COST_RATE / 2)
        )

        realized_pnl = net_value - entry_cost_basis
        realized_pnl_pct = (
            (net_value / entry_cost_basis) - 1
        ) * 100

        update_position = text("""
            UPDATE portfolio_positions
            SET
                status = 'CLOSED',
                current_price = :exit_price,
                market_value = 0,
                unrealized_pnl_inr = 0,
                unrealized_pnl_pct = 0,
                exit_date = :exit_date,
                exit_price = :exit_price,
                realized_pnl_inr = :realized_pnl,
                realized_pnl_pct = :realized_pnl_pct,
                exit_reason = :exit_reason,
                updated_at = NOW()
            WHERE id = :position_id;
        """)

        insert_transaction = text("""
            INSERT INTO portfolio_transactions (
                portfolio_position_id,
                portfolio_name,
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
            )
            VALUES (
                :portfolio_position_id,
                :portfolio_name,
                :transaction_date,
                :symbol,
                'SELL',
                :quantity,
                :price,
                :gross_value,
                :estimated_cost,
                :net_value,
                :reason,
                'exit_decision_engine'
            );
        """)

        with self.db.engine.begin() as connection:
            connection.execute(
                update_position,
                {
                    "position_id": int(position["id"]),
                    "exit_date": exit_date,
                    "exit_price": effective_exit_price,
                    "realized_pnl": realized_pnl,
                    "realized_pnl_pct": realized_pnl_pct,
                    "exit_reason": exit_reason,
                },
            )

            connection.execute(
                insert_transaction,
                {
                    "portfolio_position_id": int(position["id"]),
                    "portfolio_name": self.portfolio_name,
                    "transaction_date": exit_date,
                    "symbol": position["symbol"],
                    "quantity": quantity,
                    "price": effective_exit_price,
                    "gross_value": gross_value,
                    "estimated_cost": estimated_cost,
                    "net_value": net_value,
                    "reason": exit_reason,
                },
            )

        logger.info(
            "Closed %s | reason=%s | P&L=₹%.2f (%.2f%%)",
            position["symbol"],
            exit_reason,
            realized_pnl,
            realized_pnl_pct,
        )

    def run(self) -> pd.DataFrame:
        """Evaluate all open positions and close those requiring exit."""
        positions = self.get_open_positions()

        if positions.empty:
            logger.info("No open paper positions to evaluate.")
            return pd.DataFrame()

        market_regime = self.get_market_regime()
        closed = []

        for _, position in positions.iterrows():
            latest = self.get_latest_close(position["symbol"])

            if latest is None:
                logger.warning(
                    "%s skipped: no latest close.",
                    position["symbol"],
                )
                continue

            exit_date, current_price = latest

            exit_reason = self.get_exit_reason(
                position=position,
                current_price=current_price,
                market_regime=market_regime,
            )

            if exit_reason is None:
                continue

            self.close_position(
                position=position,
                exit_date=exit_date,
                exit_price=current_price,
                exit_reason=exit_reason,
            )

            closed.append(
                {
                    "symbol": position["symbol"],
                    "exit_date": exit_date,
                    "exit_reason": exit_reason,
                    "exit_price": current_price,
                }
            )

        return pd.DataFrame(closed)


def main() -> None:
    db = PostgresDatabase()
    db.create_schema()

    engine = ExitDecisionEngine(db=db)
    closed = engine.run()

    print("=" * 80)
    print("PAPER PORTFOLIO EXIT ENGINE")
    print("=" * 80)
    print(f"Positions exited: {len(closed)}")

    if not closed.empty:
        print(closed.to_string(index=False))


if __name__ == "__main__":
    main()