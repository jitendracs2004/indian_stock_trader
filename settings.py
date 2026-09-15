"""
Configuration for Indian Stock Trading System
Optimized for NSE/BSE with INR currency
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from nifty500_symbols import get_nifty500_symbols

load_dotenv()

class Config:

        # PostgreSQL configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "personal")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
    POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public")

    # SQLAlchemy database URL.
    # psycopg is the PostgreSQL v3 driver.
    DATABASE_URL = (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    # Market Configuration
    EXCHANGE = "NSE"  # or "BSE"
    CURRENCY = "INR"
    MARKET_OPEN = "09:15:00"
    MARKET_CLOSE = "15:30:00"

    # Data Sources (Indian market specific)
    YAHOO_FINANCE_BASE = "https://in.finance.yahoo.com"
    NSE_API_BASE = "https://www.nseindia.com"

    # Trading Parameters
    INITIAL_CAPITAL = 1000000  # INR 10 lakhs
    MAX_POSITION_SIZE = 0.10  # 10% of capital per stock
    MAX_PORTFOLIO_EXPOSURE = 0.80  # 80% max invested
    TRANSACTION_COST = 0.0003  # 0.03% (brokerage + STT + charges)
    SLIPPAGE = 0.0005  # 0.05% slippage assumption

    # Risk Management
    MAX_DRAWDOWN = 0.15  # 15% max drawdown before stopping
    VOLATILITY_TARGET = 0.20  # 20% annualized volatility target
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT = 0.10  # 10% take profit

    # Model Parameters
    LOOKBACK_WEEKLY = 20  # 20 days for weekly prediction
    LOOKBACK_MONTHLY = 60  # 60 days for monthly prediction
    TRAIN_TEST_SPLIT = 0.8
    WALK_FORWARD_WINDOWS = 5

    # File Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_DEBUG = False

    # Monitoring
    METRICS_PORT = 9090
    GRAFANA_PORT = 3000

    # Symbols to trade (Nifty 50 example)
    NIFTY_50_SYMBOLS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "SUNPHARMA.NS"
    ]

    # Full Nifty 500 universe (Yahoo Finance ".NS" tickers).
    # Sourced from nifty500_symbols.py so the list lives in one place.
    NIFTY_500_SYMBOLS = get_nifty500_symbols()

    # Sector mapping for diversification
    SECTOR_MAPPING = {
        "RELIANCE.NS": "Energy",
        "TCS.NS": "IT",
        "HDFCBANK.NS": "Banking",
        "INFY.NS": "IT",
        "ICICIBANK.NS": "Banking",
        "HINDUNILVR.NS": "FMCG",
        "ITC.NS": "FMCG",
        "SBIN.NS": "Banking",
        "BHARTIARTL.NS": "Telecom",
        "BAJFINANCE.NS": "Finance",
    }

    # -----------------------------------------------------------------
    # Entry decision and portfolio risk rules
    # -----------------------------------------------------------------

    # Weekly prediction must exceed this forecast return to be considered.
    # 1.0% is an initial conservative value, not a proven final threshold.
    WEEKLY_MIN_PREDICTED_RETURN = 0.010

    # Monthly prediction must not be materially bearish.
    MONTHLY_MIN_PREDICTED_RETURN = -0.005

    # Estimated all-in round trip cost: brokerage, taxes/charges,
    # spread/slippage approximation. Calibrate with your broker's charges.
    ESTIMATED_ROUND_TRIP_COST = 0.0016

    # Your observed early weekly MAE was 1.92 percentage points.
    # Do not treat this as a permanent figure—calculate it from walk-forward tests.
    WEEKLY_ERROR_BUFFER = 0.0192

    # Stronger, practical entry test:
    # Prediction must cover cost AND a conservative portion of forecast error.
    ERROR_BUFFER_MULTIPLIER = 0.25

    # Trend filter
    REQUIRE_CLOSE_ABOVE_SMA_50 = True

    # Market regime:
    # "not strongly bearish" means Nifty 50 / Nifty 500 index close
    # must be above its 200-day SMA.
    REGIME_SMA_SHORT = 50
    REGIME_SMA_LONG = 200

    # Liquidity: ₹5 crore average daily traded value over the past 20 sessions.
    MIN_AVG_TRADED_VALUE_INR = 5_00_00_000
    LIQUIDITY_LOOKBACK_DAYS = 20

    # Position sizing
    MAX_RISK_PER_TRADE_PCT = 0.0075      # 0.75% of capital at stop
    MAX_POSITION_VALUE_PCT = 0.10        # no more than 10% in one stock
    MAX_PORTFOLIO_EXPOSURE_PCT = 0.70     # no more than 70% total invested
    MAX_SECTOR_EXPOSURE_PCT = 0.25        # no more than 25% in one sector
    MAX_OPEN_POSITIONS = 8

    # ATR-derived initial stop
    ATR_STOP_MULTIPLIER = 2.0
    MIN_SHARES_PER_TRADE = 1

    # A raw prediction does not create an order. This must remain True
    # while developing and paper trading.
    PAPER_TRADING_ONLY = True

# Initialize config
config = Config()
