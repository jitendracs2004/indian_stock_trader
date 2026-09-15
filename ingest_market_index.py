from database import PostgresDatabase
from data_ingestion import IndianMarketData


def main():
    db = PostgresDatabase()
    db.create_schema()

    data_handler = IndianMarketData(
        data_dir="data",
        use_database=False,
    )

    index_df = data_handler.fetch_stock_data(
        symbol="^NSEI",
        period="5y",
        interval="1d",
    )

    if index_df.empty:
        raise RuntimeError("No Nifty 50 index data returned.")

    db.upsert_market_data(index_df)

    print(
        f"Stored {len(index_df)} Nifty 50 rows in market_ohlcv."
    )


if __name__ == "__main__":
    main()