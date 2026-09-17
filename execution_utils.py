"""
Execution utilities for paper trading.

These utilities prevent same-close execution bias:

Signal generated after completed daily data on date T:
    - Long entry is simulated using the first available market open
      strictly after T.
    - Long exit is simulated using the first available market open
      strictly after the exit signal date.
"""

from typing import Optional

import pandas as pd
from sqlalchemy import text

from database import PostgresDatabase


class PaperExecutionService:
    """
    Read next available daily OHLC prices and apply paper-trading slippage.

    No broker orders are sent by this class.
    """

    def __init__(
        self,
        db: PostgresDatabase,
        buy_slippage_rate: float = 0.0005,
        sell_slippage_rate: float = 0.0005,
    ):
        self.db = db
        self.buy_slippage_rate = float(buy_slippage_rate)
        self.sell_slippage_rate = float(sell_slippage_rate)

    def get_next_session_open(
        self,
        symbol: str,
        after_date: pd.Timestamp,
    ) -> Optional[dict]:
        """
        Get the first market session strictly after after_date.

        The strict > condition is essential:
        a signal based on T close cannot execute at T open/close.

        Returns:
            {
                'trade_date': pd.Timestamp,
                'open': float,
                'high': float,
                'low': float,
                'close': float
            }

        Returns None if the next session data is not available yet.
        """
        query = text("""
            SELECT
                trade_date,
                open,
                high,
                low,
                close
            FROM market_ohlcv
            WHERE symbol = :symbol
              AND trade_date > :after_date
            ORDER BY trade_date ASC
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            row = connection.execute(
                query,
                {
                    "symbol": symbol,
                    "after_date": pd.Timestamp(after_date).to_pydatetime(),
                },
            ).fetchone()

        if row is None:
            return None

        return {
            "trade_date": pd.Timestamp(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
        }

    def get_long_entry_fill(
        self,
        symbol: str,
        signal_date: pd.Timestamp,
    ) -> Optional[dict]:
        """
        Simulate a long paper entry at the next session's open plus slippage.
        """
        bar = self.get_next_session_open(
            symbol=symbol,
            after_date=signal_date,
        )

        if bar is None:
            return None

        raw_open = bar["open"]

        return {
            "fill_date": bar["trade_date"],
            "raw_open": raw_open,
            "fill_price": raw_open * (1 + self.buy_slippage_rate),
            "slippage_rate": self.buy_slippage_rate,
            "bar_high": bar["high"],
            "bar_low": bar["low"],
            "bar_close": bar["close"],
        }

    def get_long_exit_fill(
        self,
        symbol: str,
        signal_date: pd.Timestamp,
    ) -> Optional[dict]:
        """
        Simulate a long paper exit at the next session's open minus slippage.
        """
        bar = self.get_next_session_open(
            symbol=symbol,
            after_date=signal_date,
        )

        if bar is None:
            return None

        raw_open = bar["open"]

        return {
            "fill_date": bar["trade_date"],
            "raw_open": raw_open,
            "fill_price": raw_open * (1 - self.sell_slippage_rate),
            "slippage_rate": self.sell_slippage_rate,
            "bar_high": bar["high"],
            "bar_low": bar["low"],
            "bar_close": bar["close"],
        }

    def get_same_day_stop_fill(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
        stop_price: float,
    ) -> Optional[dict]:
        """
        Conservative daily-bar stop simulation for an existing long position.

        Rules:
        - If next session opens below stop: fill at open less sell slippage.
        - Else if day's low reaches stop: fill at stop less sell slippage.
        - Else: no stop fill.

        This is still EOD simulation. Intraday tick data would be needed
        for exact ordering when multiple events occur within one bar.
        """
        query = text("""
            SELECT
                trade_date,
                open,
                high,
                low,
                close
            FROM market_ohlcv
            WHERE symbol = :symbol
              AND trade_date = :trade_date
            LIMIT 1;
        """)

        with self.db.engine.connect() as connection:
            row = connection.execute(
                query,
                {
                    "symbol": symbol,
                    "trade_date": pd.Timestamp(trade_date).to_pydatetime(),
                },
            ).fetchone()

        if row is None:
            return None

        bar_date = pd.Timestamp(row[0])
        bar_open = float(row[1])
        bar_high = float(row[2])
        bar_low = float(row[3])
        bar_close = float(row[4])

        if bar_open <= stop_price:
            raw_price = bar_open
            fill_reason = "STOP_GAP_AT_OPEN"

        elif bar_low <= stop_price:
            raw_price = float(stop_price)
            fill_reason = "STOP_TOUCHED_INTRADAY"

        else:
            return None

        return {
            "fill_date": bar_date,
            "raw_price": raw_price,
            "fill_price": raw_price * (1 - self.sell_slippage_rate),
            "fill_reason": fill_reason,
            "bar_open": bar_open,
            "bar_high": bar_high,
            "bar_low": bar_low,
            "bar_close": bar_close,
        }