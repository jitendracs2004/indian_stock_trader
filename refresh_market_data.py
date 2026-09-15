"""
Incremental daily refresh for NSE equity OHLCV and Nifty 50 regime data.

What it does:
- Reads active Nifty 500 Yahoo symbols from PostgreSQL.
- Downloads a short overlap period for each symbol.
- Upserts daily OHLCV into market_ohlcv.
- Refreshes ^NSEI for the market-regime filter.
- Writes a CSV failure report if any symbols fail.

Run manually:
    python refresh_market_data.py
"""

import logging
import time
from datetime import datetime

import pandas as pd

from database import PostgresDatabase
from data_ingestion import IndianMarketData


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Short overlap is intentional. It allows corrections to the latest
# market candle and handles NSE holidays/weekends naturally.
REFRESH_PERIOD = "10d"

# Start with 25 for reliability. Increase only after stable operation.
BATCH_SIZE = 25

# Free/public data providers can rate-limit repeated requests.
SLEEP_BETWEEN_SYMBOLS_SECONDS = 0.25
SLEEP_BETWEEN_BATCHES_SECONDS = 3

# Benchmark used by entry_decision_engine.py.
MARKET_INDEX_SYMBOL = "^NSEI"

# Optional: keep None to refresh all active symbols.
# Set e.g. 50 for testing.
MAX_SYMBOLS_FOR_THIS_RUN = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_refresh_symbols(db: PostgresDatabase) -> list[str]:
    """
    Read active Nifty 500 Yahoo symbols.

    Falls back to current symbols stored in market_ohlcv if the
    nifty500_constituents table or method is not yet available.
    """
    try:
        symbols = db.get_active_nifty500_symbols()

        if symbols:
            logger.info(
                "Loaded %s active symbols from nifty500_constituents.",
                len(symbols),
            )
            return symbols

    except AttributeError:
        logger.warning(
            "get_active_nifty500_symbols() does not exist yet. "
            "Using symbols already present in market_ohlcv."
        )

    except Exception as exc:
        logger.warning(
            "Could not read nifty500_constituents: %s. "
            "Using symbols already present in market_ohlcv.",
            exc,
        )

    from sqlalchemy import text

    query = text("""
        SELECT DISTINCT symbol
        FROM market_ohlcv
        WHERE symbol NOT LIKE '^%%'
          AND symbol NOT IN ('NIFTY50.NS', 'NIFTY500.NS')
        ORDER BY symbol;
    """)

    with db.engine.connect() as connection:
        rows = connection.execute(query).fetchall()

    return [row[0] for row in rows]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data returned by data_ingestion.py."""
    if df.empty:
        return df

    data = df.copy()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    if "datetime" in data.columns and "date" not in data.columns:
        data = data.rename(columns={"datetime": "date"})

    expected = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in expected
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}. "
            f"Received: {data.columns.tolist()}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        ).astype("float64")

    # Convert timezone-aware Yahoo timestamps to timezone-naive values
    # for consistent PostgreSQL trade_date storage.
    if getattr(data["date"].dt, "tz", None) is not None:
        data["date"] = data["date"].dt.tz_localize(None)

    data = data.dropna(
        subset=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return data


def refresh_one_symbol(
    data_handler: IndianMarketData,
    db: PostgresDatabase,
    symbol: str,
) -> int:
    """Download one symbol's short history and upsert it."""
    raw_df = data_handler.fetch_stock_data(
        symbol=symbol,
        period=REFRESH_PERIOD,
        interval="1d",
    )

    if raw_df.empty:
        raise RuntimeError("No market data returned.")

    clean_df = clean_ohlcv(raw_df)

    if clean_df.empty:
        raise RuntimeError("No valid data after OHLCV cleaning.")

    return db.upsert_market_data(clean_df)


def main() -> None:
    started_at = datetime.now()

    print("=" * 80)
    print("INCREMENTAL NSE MARKET-DATA REFRESH")
    print("=" * 80)
    print(f"Started: {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"Refresh period: {REFRESH_PERIOD}")
    print("=" * 80)

    db = PostgresDatabase()

    if not db.test_connection():
        raise ConnectionError(
            "PostgreSQL connection failed. "
            "Check .env and local PostgreSQL service."
        )

    db.create_schema()

    data_handler = IndianMarketData(data_dir="data")

    # Refresh benchmark first so entry regime uses recent data.
    try:
        index_rows = refresh_one_symbol(
            data_handler=data_handler,
            db=db,
            symbol=MARKET_INDEX_SYMBOL,
        )

        logger.info(
            "Refreshed %s: %s rows.",
            MARKET_INDEX_SYMBOL,
            index_rows,
        )

    except Exception as exc:
        logger.exception(
            "Could not refresh market index %s: %s",
            MARKET_INDEX_SYMBOL,
            exc,
        )

        # Do not exit here. Stock data may still refresh, but entry engine
        # should later block entries if regime is unavailable.

    symbols = get_refresh_symbols(db)

    if not symbols:
        raise RuntimeError(
            "No equity symbols are available to refresh."
        )

    symbols = sorted(set(symbols))

    if MAX_SYMBOLS_FOR_THIS_RUN is not None:
        symbols = symbols[:MAX_SYMBOLS_FOR_THIS_RUN]

    success_rows = []
    failure_rows = []
    total_rows_upserted = 0

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]

        logger.info(
            "Processing batch %s: symbols %s to %s of %s",
            start // BATCH_SIZE + 1,
            start + 1,
            min(start + BATCH_SIZE, len(symbols)),
            len(symbols),
        )

        for symbol in batch:
            try:
                upserted = refresh_one_symbol(
                    data_handler=data_handler,
                    db=db,
                    symbol=symbol,
                )

                success_rows.append({
                    "symbol": symbol,
                    "rows_upserted": upserted,
                })

                total_rows_upserted += upserted

                logger.info(
                    "%s refreshed: %s rows.",
                    symbol,
                    upserted,
                )

            except Exception as exc:
                logger.exception(
                    "%s refresh failed: %s",
                    symbol,
                    exc,
                )

                failure_rows.append({
                    "symbol": symbol,
                    "reason": str(exc),
                })

            time.sleep(SLEEP_BETWEEN_SYMBOLS_SECONDS)

        if start + BATCH_SIZE < len(symbols):
            time.sleep(SLEEP_BETWEEN_BATCHES_SECONDS)

    today = datetime.now().strftime("%Y%m%d")

    success_df = pd.DataFrame(success_rows)
    failure_df = pd.DataFrame(failure_rows)

    if not success_df.empty:
        success_df.to_csv(
            f"refresh_success_{today}.csv",
            index=False,
        )

    if not failure_df.empty:
        failure_df.to_csv(
            f"refresh_failures_{today}.csv",
            index=False,
        )

    print("\n" + "=" * 80)
    print("MARKET-DATA REFRESH COMPLETE")
    print("=" * 80)
    print(f"Symbols attempted: {len(symbols)}")
    print(f"Successful:        {len(success_rows)}")
    print(f"Failed:            {len(failure_rows)}")
    print(f"Rows upserted:     {total_rows_upserted:,}")
    print(
        f"Finished:          {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    if not failure_df.empty:
        print(
            f"Failure report: refresh_failures_{today}.csv"
        )


if __name__ == "__main__":
    main()