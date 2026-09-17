"""
Paper Portfolio Engine.

Responsibilities:
- Read ELIGIBLE_BUY rows from entry_decisions.
- Open paper positions only when no open position exists.
- Use decision-engine quantity, entry price, stop, sector.
- Track current mark-to-market P&L from market_ohlcv.
- Save daily portfolio snapshots.

This module does NOT send broker orders.
"""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from database import PostgresDatabase
from settings import config

from execution_utils import PaperExecutionService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

PORTFOLIO_NAME = "paper_default"

# Conservative paper-trading cost estimate.
# Replace later with actual broker/STT/exchange/GST/stamp calculations.
PAPER_TRADE_COST_RATE = 0.0016


class PaperPortfolioEngine:
    """Create and manage paper positions."""

    def __init__(
        self,
        db: PostgresDatabase,
        portfolio_name: str = PORTFOLIO_NAME,
        initial_capital: float | None = None,
    ):
        self.db = db
        self.portfolio_name = portfolio_name
        self.initial_capital = float(
            initial_capital or config.INITIAL_CAPITAL
        )
        self.execution = PaperExecutionService(
            db=self.db,
            buy_slippage_rate=PAPER_TRADE_COST_RATE / 2,
            sell_slippage_rate=PAPER_TRADE_COST_RATE / 2,
        )

    def get_open_positions(self) -> pd.DataFrame:
        """Return currently open paper positions."""
        query = text("""
            SELECT
                id,
                portfolio_name,
                symbol,
                sector,
                entry_decision_id,
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
            ORDER BY entry_date ASC, id ASC;
        """)

        with self.db.engine.connect() as connection:
            df = pd.read_sql(
                query,
                connection,
                params={"portfolio_name": self.portfolio_name},
            )

        return df

    def get_latest_price(self, symbol: str) -> tuple[datetime, float] | None:
        """Get latest available close from market_ohlcv."""
        query = text("""
            SELECT
                trade_date,
                close
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

    def get_available_cash(self) -> float:
        """
        Calculate available paper cash:
        initial capital + closed-sale proceeds - open buy costs.

        This simple cash ledger avoids relying on a separate mutable balance.
        """
        query = text("""
            SELECT
                COALESCE(
                    SUM(
                        CASE
                            WHEN transaction_type = 'SELL'
                            THEN net_value
                            WHEN transaction_type = 'BUY'
                            THEN -net_value
                            ELSE 0
                        END
                    ),
                    0
                ) AS transaction_cash_flow
            FROM portfolio_transactions
            WHERE portfolio_name = :portfolio_name;
        """)

        with self.db.engine.connect() as connection:
            cash_flow = connection.execute(
                query,
                {"portfolio_name": self.portfolio_name},
            ).scalar()

        return self.initial_capital + float(cash_flow or 0)

    def get_latest_eligible_decisions(self) -> pd.DataFrame:
        """
        Get one latest ELIGIBLE_BUY decision per stock.

        Decisions that have already become positions are excluded later.
        """
        query = text("""
            SELECT DISTINCT ON (symbol)
                id AS entry_decision_id,
                symbol,
                decision_date,
                sector,
                entry_status,
                entry_price,
                initial_stop_price,
                suggested_quantity,
                suggested_position_value,
                weekly_predicted_return,
                monthly_predicted_return,
                market_regime,
                model_name,
                model_version,
                created_at
            FROM entry_decisions
            WHERE entry_status = 'ELIGIBLE_BUY'
            ORDER BY
                symbol,
                decision_date DESC,
                created_at DESC,
                id DESC;
        """)

        with self.db.engine.connect() as connection:
            return pd.read_sql(query, connection)

    def open_new_positions(self) -> pd.DataFrame:
        """
        Create paper positions from current ELIGIBLE_BUY decisions.

        Timing policy:
        - Decision is based on completed EOD data at decision_date.
        - Fill occurs at the next available NSE daily open.
        - If next-session data is not available yet, position remains
        unfilled and will be evaluated during a future pipeline run.
        """
        decisions = self.get_latest_eligible_decisions()

        if decisions.empty:
            logger.info("No ELIGIBLE_BUY decisions available.")
            return pd.DataFrame()

        open_positions = self.get_open_positions()
        open_symbols = set(open_positions["symbol"].astype(str))

        available_cash = self.get_available_cash()
        created = []

        for _, decision in decisions.iterrows():
            symbol = str(decision["symbol"])

            if symbol in open_symbols:
                logger.info(
                    "%s skipped: already open in paper portfolio.",
                    symbol,
                )
                continue

            suggested_quantity = int(
                decision["suggested_quantity"] or 0
            )

            if suggested_quantity <= 0:
                logger.info(
                    "%s skipped: no valid suggested quantity.",
                    symbol,
                )
                continue

            decision_date = pd.Timestamp(
                decision["decision_date"]
            )

            # Critical correction:
            # Fill is at NEXT available session open, not decision-date close.
            fill = self.execution.get_long_entry_fill(
                symbol=symbol,
                signal_date=decision_date,
            )

            if fill is None:
                logger.info(
                    "%s: entry pending; next trading-session open is not "
                    "available after decision date %s.",
                    symbol,
                    decision_date.date(),
                )
                continue

            entry_date = fill["fill_date"]
            entry_price = float(fill["fill_price"])

            # The decision engine calculated a stop based on signal close.
            # Preserve the same percentage distance when next-open fill differs.
            decision_reference_price = float(
                decision["entry_price"]
            )

            decision_stop = float(
                decision["initial_stop_price"]
            )

            stop_distance_pct = (
                (decision_reference_price - decision_stop)
                / decision_reference_price
            )

            initial_stop = entry_price * (
                1 - stop_distance_pct
            )

            # Size is recomputed using actual next-session entry price while
            # preserving max position value through the suggested quantity cap.
            max_risk_inr = (
                self.initial_capital
                * config.MAX_RISK_PER_TRADE_PCT
            )

            risk_per_share = entry_price - initial_stop

            if risk_per_share <= 0:
                logger.warning(
                    "%s skipped: invalid stop/risk after next-open fill.",
                    symbol,
                )
                continue

            quantity_by_risk = int(
                max_risk_inr / risk_per_share
            )

            quantity = min(
                suggested_quantity,
                quantity_by_risk,
            )

            if quantity <= 0:
                logger.info(
                    "%s skipped: risk-based quantity is zero.",
                    symbol,
                )
                continue

            gross_value = entry_price * quantity
            estimated_cost = gross_value * (
                PAPER_TRADE_COST_RATE / 2
            )
            net_value = gross_value + estimated_cost

            # Reduce quantity if actual next-session open made it too expensive.
            if net_value > available_cash:
                quantity = int(
                    available_cash
                    / (
                        entry_price
                        * (1 + PAPER_TRADE_COST_RATE / 2)
                    )
                )

                if quantity <= 0:
                    logger.info(
                        "%s skipped: insufficient cash at next-session open.",
                        symbol,
                    )
                    continue

                gross_value = entry_price * quantity
                estimated_cost = gross_value * (
                    PAPER_TRADE_COST_RATE / 2
                )
                net_value = gross_value + estimated_cost

            insert_position = text("""
                INSERT INTO portfolio_positions (
                    portfolio_name,
                    symbol,
                    sector,
                    status,
                    entry_decision_id,
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
                )
                VALUES (
                    :portfolio_name,
                    :symbol,
                    :sector,
                    'OPEN',
                    :entry_decision_id,
                    :entry_date,
                    :entry_price,
                    :quantity,
                    :initial_stop_price,
                    :trailing_stop_price,
                    :highest_price,
                    :current_price,
                    :market_value,
                    :unrealized_pnl_inr,
                    :unrealized_pnl_pct
                )
                RETURNING id;
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
                    'BUY',
                    :quantity,
                    :price,
                    :gross_value,
                    :estimated_cost,
                    :net_value,
                    :reason,
                    'paper_portfolio_engine'
                );
            """)

            with self.db.engine.begin() as connection:
                position_id = connection.execute(
                    insert_position,
                    {
                        "portfolio_name": self.portfolio_name,
                        "symbol": symbol,
                        "sector": decision["sector"] or "UNKNOWN",
                        "entry_decision_id": int(
                            decision["entry_decision_id"]
                        ),
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "initial_stop_price": initial_stop,
                        "trailing_stop_price": initial_stop,
                        "highest_price": entry_price,
                        "current_price": entry_price,
                        "market_value": gross_value,
                        "unrealized_pnl_inr": -estimated_cost,
                        "unrealized_pnl_pct": (
                            -estimated_cost / gross_value
                        ) * 100,
                    },
                ).scalar_one()

                connection.execute(
                    insert_transaction,
                    {
                        "portfolio_position_id": position_id,
                        "portfolio_name": self.portfolio_name,
                        "transaction_date": entry_date,
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": entry_price,
                        "gross_value": gross_value,
                        "estimated_cost": estimated_cost,
                        "net_value": net_value,
                        "reason": (
                            "ELIGIBLE_BUY_NEXT_SESSION_OPEN"
                        ),
                    },
                )

            available_cash -= net_value
            open_symbols.add(symbol)

            created.append(
                {
                    "position_id": position_id,
                    "symbol": symbol,
                    "signal_date": decision_date,
                    "entry_date": entry_date,
                    "raw_open": fill["raw_open"],
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "initial_stop_price": initial_stop,
                    "position_value": gross_value,
                }
            )

            logger.info(
                "Opened paper position | %s | signal=%s | fill=%s | "
                "open=%.2f | entry=%.2f | qty=%s | stop=%.2f",
                symbol,
                decision_date.date(),
                entry_date.date(),
                fill["raw_open"],
                entry_price,
                quantity,
                initial_stop,
            )

        return pd.DataFrame(created)

    def mark_to_market(self) -> pd.DataFrame:
        """Update open positions using latest close data."""
        positions = self.get_open_positions()

        if positions.empty:
            return positions

        updated_rows = []

        update_sql = text("""
            UPDATE portfolio_positions
            SET
                highest_price_since_entry = :highest_price,
                current_price = :current_price,
                market_value = :market_value,
                unrealized_pnl_inr = :unrealized_pnl_inr,
                unrealized_pnl_pct = :unrealized_pnl_pct,
                updated_at = NOW()
            WHERE id = :position_id;
        """)

        for _, position in positions.iterrows():
            latest = self.get_latest_price(position["symbol"])

            if latest is None:
                logger.warning(
                    "Cannot mark %s: no latest close.",
                    position["symbol"],
                )
                continue

            price_date, current_price = latest

            quantity = int(position["quantity"])
            entry_price = float(position["entry_price"])

            highest_price = max(
                float(position["highest_price_since_entry"] or entry_price),
                current_price,
            )

            market_value = current_price * quantity
            unrealized_pnl_inr = (
                current_price - entry_price
            ) * quantity

            unrealized_pnl_pct = (
                (current_price / entry_price) - 1
            ) * 100

            with self.db.engine.begin() as connection:
                connection.execute(
                    update_sql,
                    {
                        "position_id": int(position["id"]),
                        "highest_price": highest_price,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pnl_inr": unrealized_pnl_inr,
                        "unrealized_pnl_pct": unrealized_pnl_pct,
                    },
                )

            updated_rows.append(
                {
                    "position_id": int(position["id"]),
                    "symbol": position["symbol"],
                    "price_date": price_date,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl_inr": unrealized_pnl_inr,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                }
            )

        return pd.DataFrame(updated_rows)

    def create_daily_snapshot(self) -> dict:
        """Save one daily portfolio snapshot."""
        positions = self.get_open_positions()
        cash = self.get_available_cash()

        holdings_value = (
            float(positions["market_value"].fillna(0).sum())
            if not positions.empty
            else 0.0
        )

        total_value = cash + holdings_value
        open_positions = len(positions)

        gross_exposure_pct = (
            holdings_value / total_value
            if total_value > 0
            else 0.0
        )

        total_pnl = total_value - self.initial_capital

        previous_query = text("""
            SELECT total_value_inr
            FROM portfolio_daily_snapshots
            WHERE portfolio_name = :portfolio_name
            ORDER BY snapshot_date DESC
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            previous_total = connection.execute(
                previous_query,
                {"portfolio_name": self.portfolio_name},
            ).scalar()

        daily_pnl = (
            total_value - float(previous_total)
            if previous_total is not None
            else total_pnl
        )

        peak_query = text("""
            SELECT MAX(total_value_inr)
            FROM portfolio_daily_snapshots
            WHERE portfolio_name = :portfolio_name;
        """)

        with self.db.engine.connect() as connection:
            historical_peak = connection.execute(
                peak_query,
                {"portfolio_name": self.portfolio_name},
            ).scalar()

        peak_value = max(
            self.initial_capital,
            float(historical_peak or 0),
            total_value,
        )

        drawdown_pct = (
            (total_value / peak_value - 1) * 100
            if peak_value > 0
            else 0.0
        )

        snapshot_date = datetime.now().date()

        upsert_snapshot = text("""
            INSERT INTO portfolio_daily_snapshots (
                portfolio_name,
                snapshot_date,
                cash_inr,
                holdings_value_inr,
                total_value_inr,
                open_positions,
                gross_exposure_pct,
                daily_pnl_inr,
                total_pnl_inr,
                drawdown_pct
            )
            VALUES (
                :portfolio_name,
                :snapshot_date,
                :cash_inr,
                :holdings_value_inr,
                :total_value_inr,
                :open_positions,
                :gross_exposure_pct,
                :daily_pnl_inr,
                :total_pnl_inr,
                :drawdown_pct
            )
            ON CONFLICT (portfolio_name, snapshot_date)
            DO UPDATE SET
                cash_inr = EXCLUDED.cash_inr,
                holdings_value_inr = EXCLUDED.holdings_value_inr,
                total_value_inr = EXCLUDED.total_value_inr,
                open_positions = EXCLUDED.open_positions,
                gross_exposure_pct = EXCLUDED.gross_exposure_pct,
                daily_pnl_inr = EXCLUDED.daily_pnl_inr,
                total_pnl_inr = EXCLUDED.total_pnl_inr,
                drawdown_pct = EXCLUDED.drawdown_pct,
                created_at = NOW();
        """)

        snapshot = {
            "portfolio_name": self.portfolio_name,
            "snapshot_date": snapshot_date,
            "cash_inr": cash,
            "holdings_value_inr": holdings_value,
            "total_value_inr": total_value,
            "open_positions": open_positions,
            "gross_exposure_pct": gross_exposure_pct,
            "daily_pnl_inr": daily_pnl,
            "total_pnl_inr": total_pnl,
            "drawdown_pct": drawdown_pct,
        }

        with self.db.engine.begin() as connection:
            connection.execute(upsert_snapshot, snapshot)

        return snapshot


def main() -> None:
    db = PostgresDatabase()
    db.create_schema()

    engine = PaperPortfolioEngine(db=db)

    print("=" * 80)
    print("PAPER PORTFOLIO ENGINE")
    print("=" * 80)

    marked = engine.mark_to_market()
    print(f"Positions marked to market: {len(marked)}")

    created = engine.open_new_positions()
    print(f"New paper positions opened: {len(created)}")

    # Mark again to include newly created rows in snapshot.
    engine.mark_to_market()

    snapshot = engine.create_daily_snapshot()

    print(f"Cash:           ₹{snapshot['cash_inr']:,.2f}")
    print(f"Holdings:       ₹{snapshot['holdings_value_inr']:,.2f}")
    print(f"Portfolio:      ₹{snapshot['total_value_inr']:,.2f}")
    print(f"Open positions: {snapshot['open_positions']}")
    print(f"Drawdown:       {snapshot['drawdown_pct']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()