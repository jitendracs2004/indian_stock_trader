"""
Data Ingestion Module for Indian Stock Market
Supports Yahoo Finance India, NSE data, and CSV imports
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional
import os
from database import PostgresDatabase
from nifty500_symbols import NIFTY_500_SYMBOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianMarketData:
    """
    Fetch and manage Indian stock market data from Yahoo Finance India
    """

    def __init__(
        self,
        data_dir: str = "data",
        use_database: bool = True,
    ):
        self.data_dir = data_dir
        self.use_database = use_database

        os.makedirs(data_dir, exist_ok=True)

        self.db = PostgresDatabase() if use_database else None

        if self.db is not None:
            if not self.db.test_connection():
                raise ConnectionError(
                    "Could not connect to PostgreSQL. "
                    "Check POSTGRES_* values in .env."
                )

            self.db.create_schema()

    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data for Indian stocks

        Args:
            symbol: NSE symbol with .NS suffix (e.g., "RELIANCE.NS")
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Move the yfinance DatetimeIndex / Date index into a normal column first
            df = df.reset_index()

            # Normalize every column name after reset_index()
            df.columns = [str(col).strip().lower() for col in df.columns]

            # yfinance may call the timestamp column either "date" or "datetime".
            # Standardize it so the rest of this project always uses "date".
            if "datetime" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"datetime": "date"})

            if "date" not in df.columns:
                raise ValueError(
                    f"Could not find a date column for {symbol}. Columns received: {df.columns.tolist()}"
                )

            # Ensure a standard Pandas datetime column; daily data should normally be timezone-naive.
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if getattr(df["date"].dt, "tz", None) is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            # Add symbol column
            df["symbol"] = symbol

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_stocks(self, symbols: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch data for multiple stocks and combine

        Args:
            symbols: List of NSE symbols
            period: Time period

        Returns:
            Combined DataFrame
        """
        all_data = []

        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            df = self.fetch_stock_data(symbol, period)
            if not df.empty:
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Fetched {len(combined)} total rows for {len(symbols)} symbols")
            return combined
        else:
            logger.warning("No data fetched for any symbols")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, parse_dates=['date'] if 'date' in pd.read_csv(filepath, nrows=0).columns else [0])
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
        return None

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df.empty:
            return False

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return False

        # Check for NaN values
        if df[required_cols].isnull().sum().sum() > 0:
            logger.warning(f"Found NaN values in data")

        # Check for negative prices
        if (df[required_cols[:-1]] < 0).any().any():
            logger.error("Found negative prices in data")
            return False

        # Check volume
        if (df['volume'] < 0).any():
            logger.error("Found negative volume in data")
            return False

        return True
    
    def fetch_store_and_load(
        self,
        symbols: List[str],
        period: str = "5y",
        refresh_from_yahoo: bool = True,
    ) -> pd.DataFrame:
        """
        Optionally download data, upsert it into PostgreSQL,
        then return the canonical dataset from PostgreSQL.
        """
        if self.db is None:
            raise RuntimeError("Database is disabled.")

        if refresh_from_yahoo:
            downloaded = self.fetch_multiple_stocks(symbols, period=period)

            if not downloaded.empty:
                self.db.upsert_market_data(downloaded)

        return self.db.get_market_data(symbols=symbols)


# Example usage
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Ingest NSE OHLCV data from Yahoo Finance into PostgreSQL."
    )
    parser.add_argument(
        "--period",
        default="5y",
        help='History to fetch, e.g. "1y", "2y", "5y", "max" (default: 5y).',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N symbols (useful for a quick test run).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of symbols fetched before writing to PostgreSQL (default: 25).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to pause between batches to ease Yahoo rate limits (default: 1.0).",
    )
    args = parser.parse_args()

    # Full Nifty 500 universe, optionally sliced for a test run.
    symbols = NIFTY_500_SYMBOLS
    if args.limit is not None:
        symbols = symbols[: args.limit]

    # Uses PostgreSQL so the dashboard's load_symbols() can see the data.
    data_handler = IndianMarketData(data_dir="data", use_database=True)

    total = len(symbols)
    print(
        f"Ingesting {total} symbols into PostgreSQL "
        f"(period={args.period}, batch_size={args.batch_size})."
    )

    ingested_symbols = 0
    failed_symbols = []

    for start in range(0, total, args.batch_size):
        batch = symbols[start : start + args.batch_size]
        batch_no = (start // args.batch_size) + 1
        print(
            f"\n[Batch {batch_no}] Fetching symbols "
            f"{start + 1}-{min(start + len(batch), total)} of {total}..."
        )

        # fetch_store_and_load downloads, upserts into PostgreSQL,
        # then returns the canonical rows from the database.
        stored = data_handler.fetch_store_and_load(
            symbols=batch,
            period=args.period,
            refresh_from_yahoo=True,
        )

        found = set(stored["symbol"].unique()) if not stored.empty else set()
        for sym in batch:
            if sym in found:
                ingested_symbols += 1
            else:
                failed_symbols.append(sym)

        if start + args.batch_size < total:
            time.sleep(args.sleep)

    print("\n" + "-" * 60)
    print(f"Ingestion complete: {ingested_symbols}/{total} symbols stored.")
    if failed_symbols:
        print(f"No data returned for {len(failed_symbols)} symbols:")
        print(", ".join(failed_symbols))
    print("-" * 60)
