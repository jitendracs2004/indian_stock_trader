# Indian Stock Trading System

## Overview
Complete end-to-end algorithmic trading system for Indian stock market (NSE/BSE) with:
- **Full Nifty 500 universe** ingested from Yahoo Finance India into PostgreSQL
- **Daily, weekly, and monthly prediction** using ensemble ML models (LSTM + XGBoost)
- **Data ingestion** from Yahoo Finance India, persisted in PostgreSQL
- **40+ technical indicators** for feature engineering
- **Entry decision engine** — a multi-condition filter (trend, market regime,
  liquidity, exposure limits, ATR sizing) with a daily timing gate
- **Exit decision engine** — stop-loss, trailing stop, model-reversal, and
  time-stop rules for open paper positions
- **Paper portfolio engine** — marks positions to market, opens eligible
  buys, logs transactions, and saves daily snapshots (no live orders)
- **Streamlit dashboard** with four tabs: raw signals, entry decisions,
  per-symbol analysis, and paper portfolio

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  Yahoo Finance India → Data Ingestion → PostgreSQL          │
│  (Nifty 500 universe, ~5y daily OHLCV)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Feature Engineering                            │
│  TA-Lib Indicators (40+) + Custom Features + Targets        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  ML Models                                   │
│  LSTM (time-series) + XGBoost (tabular) → Ensemble Voting   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Decision Engines                               │
│  Entry engine (filters + daily timing gate) →               │
│  Exit engine (stops, reversals, time stops) →               │
│  Paper portfolio engine (mark-to-market, snapshots)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Presentation Layer                           │
│  Streamlit Dashboard — Raw Signals · Entry Decisions ·      │
│  Stock Analysis · Paper Portfolio                           │
└─────────────────────────────────────────────────────────────┘
```

### PostgreSQL tables

| Table | Written by | Purpose |
|-------|-----------|---------|
| `market_ohlcv` | `data_ingestion.py`, `refresh_market_data.py` | Daily OHLCV price history |
| `model_signals` | `predict_weekly_monthly.py` | Daily/weekly/monthly model predictions |
| `entry_decisions` | `entry_decision_engine.py` | Final entry verdict per symbol + inputs |
| `portfolio_positions` | `portfolio_engine.py`, `exit_decision_engine.py` | Open and closed paper positions |
| `portfolio_transactions` | `portfolio_engine.py`, `exit_decision_engine.py` | BUY/SELL log with estimated costs |
| `portfolio_daily_snapshots` | `portfolio_engine.py` | Daily portfolio value / exposure / P&L |

## Quick Start

### 1. Installation

```bash
# Clone or copy the project
cd indian_stock_trader

# Install system dependencies (TA-Lib)
# Ubuntu/Debian:
sudo apt-get install ta-lib

# macOS:
brew install ta-lib

# Windows: Download from https://github.com/cgohlke/talib-binary

# Install Python dependencies
pip install -r requirements.txt

# Or run automated setup
python setup.py
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# (optional - defaults work for most cases)
```

### 3. Run Everything (One Command)

To go from an empty database to a running dashboard in a single step,
use the setup script. It ingests the Nifty 500, trains models, writes
predictions, then launches the dashboard.

```powershell
# Full run: all Nifty 500 symbols, 5 years of history, then dashboard
.\run_full_setup.ps1

# Recommended first time — quick test with 20 symbols:
.\run_full_setup.ps1 -Limit 20

# Other options
.\run_full_setup.ps1 -Period 2y        # shorter history
.\run_full_setup.ps1 -SkipDashboard    # run ingestion + prediction only
```

When it finishes, the dashboard opens in your browser showing every
symbol that has data. Press `Ctrl+C` in the terminal to stop it.

> **First-time setup vs. daily updates.** `run_full_setup.ps1` does a
> full download and is meant for initial setup. For recurring
> end-of-day updates, use `run_daily_pipeline.ps1` (via `daily_pipeline.py`),
> which incrementally refreshes recent candles and re-runs predictions —
> much faster and lighter on Yahoo.

### 4. Run the Pipeline Manually (Step by Step)

If you'd rather run each stage yourself, or need to re-run just one:

```bash
# Stage 1 — Ingest OHLCV data into PostgreSQL.
# Test with a small slice first to confirm the pipeline works:
python data_ingestion.py --limit 20

# Then ingest the full Nifty 500 universe (this takes a while):
python data_ingestion.py --period 5y

# Stage 2 — Train models and write daily/weekly/monthly signals
# to the model_signals table and weekly_monthly_predictions.csv:
python predict_weekly_monthly.py

# Stage 3 — Run the decision engines (in this order):
python entry_decision_engine.py    # final entry verdicts
python exit_decision_engine.py     # close positions needing exit
python portfolio_engine.py         # mark positions, open buys, snapshot

# Stage 4 — Launch the dashboard (reads whatever is in PostgreSQL):
python -m streamlit run streamlit_dashboard.py
```

The `daily_pipeline.py` orchestrator runs stages 1–3 in the correct
order for scheduled end-of-day updates (see the Task Scheduler section).

#### `data_ingestion.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--period` | `5y` | History depth: `1y`, `2y`, `5y`, `max` |
| `--limit N` | all | Ingest only the first N symbols (quick test) |
| `--batch-size` | `25` | Symbols fetched per PostgreSQL write |
| `--sleep` | `1.0` | Seconds to pause between batches (rate-limit relief) |

### 5. Notes on the Nifty 500 Universe

- The constituent list lives in `nifty500_symbols.py` as Yahoo `.NS`
  tickers, exposed via `config.NIFTY_500_SYMBOLS` in `settings.py`.
- Index membership drifts over time. Update `_NIFTY_500_BASE` in
  `nifty500_symbols.py` to refresh it.
- A handful of symbols may not resolve on Yahoo (renamed or delisted
  tickers) and will be reported at the end of ingestion.
- Recent IPOs without ~300 rows of history are skipped during model
  training and logged — this is an intentional guard, not an error.

## Automated Daily Runs (Windows Task Scheduler)

Once the initial setup is done, schedule the daily pipeline to run
automatically after NSE close. The recommended schedule is **weekdays
(Mon–Fri) at 6:00 PM**, which is after the 3:30 PM market close.

The pipeline is driven by `run_daily_pipeline.ps1`, which calls
`daily_pipeline.py` and writes a timestamped log to `logs/`. The
orchestrator runs these steps in order:

1. `refresh_market_data.py` — incremental OHLCV refresh
2. `predict_weekly_monthly.py` — daily/weekly/monthly signals
3. `entry_decision_engine.py` — final entry verdicts
4. `exit_decision_engine.py` — close positions needing exit
5. `portfolio_engine.py` — mark positions, open eligible buys, snapshot

### Before you schedule

1. Confirm your Python path matches the one in `run_daily_pipeline.ps1`:

   ```powershell
   python -c "import sys; print(sys.executable)"
   ```

   If it differs, update `$PythonExecutable` at the top of
   `run_daily_pipeline.ps1`.

2. Complete the one-time full setup at least once so PostgreSQL has data
   (the daily refresh only updates symbols already in the database):

   ```powershell
   .\run_full_setup.ps1 -SkipDashboard
   ```

### Option A — Register from PowerShell (fastest)

Run this once in a PowerShell window (no admin needed for a user-level
task):

```powershell
$scriptPath = "C:\Users\elnpqtw\OneDrive - Ericsson\Documents\Jit\Personal\Learning\indian_stock_trader\run_daily_pipeline.ps1"

$action   = New-ScheduledTaskAction -Execute "powershell.exe" `
              -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger  = New-ScheduledTaskTrigger -Weekly `
              -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
              -At 6:00PM

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
              -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName "IndianStockTrader_DailyPipeline" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Runs the daily EOD pipeline on weekdays at 6 PM." -Force
```

Verify it and check the next run time:

```powershell
Get-ScheduledTaskInfo -TaskName "IndianStockTrader_DailyPipeline" |
    Select-Object TaskName, NextRunTime, LastRunTime, LastTaskResult
```

Run it once immediately to confirm it works end to end:

```powershell
Start-ScheduledTask -TaskName "IndianStockTrader_DailyPipeline"
```

Then check the newest log:

```powershell
Get-ChildItem logs\daily_pipeline_*.log | Sort-Object LastWriteTime |
    Select-Object -Last 1 | Get-Content -Tail 40
```

### Option B — Task Scheduler GUI

1. Open **Task Scheduler** (`taskschd.msc`).
2. Click **Create Task** (not "Basic Task").
3. **General** tab: name it `IndianStockTrader_DailyPipeline`.
   Select **Run whether user is logged on or not** if you want it to run
   without an active session.
4. **Triggers** tab → **New**:
   - Begin the task: **On a schedule**
   - **Weekly**, recur every **1** week
   - Check **Monday, Tuesday, Wednesday, Thursday, Friday**
   - Start time: **6:00:00 PM**
5. **Actions** tab → **New**:
   - Action: **Start a program**
   - Program/script: `powershell.exe`
   - Add arguments:
     ```
     -NoProfile -ExecutionPolicy Bypass -File "C:\Users\elnpqtw\OneDrive - Ericsson\Documents\Jit\Personal\Learning\indian_stock_trader\run_daily_pipeline.ps1"
     ```
6. **Settings** tab: enable **Run task as soon as possible after a
   scheduled start is missed** (covers the case where the machine was off
   at 6 PM).
7. Click **OK**.

### Managing the task

```powershell
# Disable temporarily
Disable-ScheduledTask -TaskName "IndianStockTrader_DailyPipeline"

# Re-enable
Enable-ScheduledTask -TaskName "IndianStockTrader_DailyPipeline"

# Remove entirely
Unregister-ScheduledTask -TaskName "IndianStockTrader_DailyPipeline" -Confirm:$false
```

> **Note:** The task only fires when the computer is on at 6 PM. If the
> machine may be asleep or off, enable the "run as soon as possible after
> a missed start" setting (shown above), and optionally check **Wake the
> computer to run this task** in the trigger's advanced settings.

## Module Structure

```
indian_stock_trader/
├── settings.py                 # System configuration
├── nifty500_symbols.py         # Nifty 500 constituent list (Yahoo tickers)
├── data_ingestion.py           # Yahoo Finance → PostgreSQL ingestion (CLI)
├── database.py                 # PostgreSQL access layer
├── feature_engineering.py      # Technical indicators (40+)
├── ml_models.py                # LSTM + XGBoost ensemble
├── predict_weekly_monthly.py   # Train models + write daily/weekly/monthly signals
├── entry_decision_engine.py    # Final entry-condition engine (+ daily timing gate)
├── exit_decision_engine.py     # Exit rules for open paper positions
├── portfolio_engine.py         # Paper portfolio: mark-to-market, open buys, snapshots
├── execution_utils.py          # Shared execution/cost helpers
├── refresh_market_data.py      # Incremental OHLCV refresh (daily)
├── daily_pipeline.py           # Daily orchestrator (refresh → predict → entry → exit → portfolio)
├── verify_last_week.py         # Backtest-style forecast verification
├── backtester.py               # Backtesting engine
├── risk_management.py          # Risk management
├── trading_engine.py           # End-to-end orchestration example
├── streamlit_dashboard.py      # Streamlit dashboard (4 tabs)
├── run_full_setup.ps1          # One-time full setup + dashboard launch
├── run_daily_pipeline.ps1      # Task Scheduler wrapper for daily_pipeline.py
├── requirements.txt            # Dependencies
├── setup.py                    # Installation script
└── README.md                   # This file
```

## Dashboard Tabs

The Streamlit dashboard (`streamlit_dashboard.py`) reads directly from
PostgreSQL and is organized into four tabs. Pick a symbol and chart
window from the sidebar.

1. **Raw Signals** — Top 10 raw model BUY signals per horizon
   (Daily / Weekly / Monthly), unfiltered by the entry rules.
2. **Entry Decisions** — the Top 10 eligible-buy shortlist that cleared
   every entry check, plus the selected symbol's final verdict with
   daily/weekly/monthly signal pills. Statuses: `ELIGIBLE_BUY`,
   `WAIT_DAILY_TIMING` (thesis ok, daily says wait), or `BLOCKED`.
3. **Stock Analysis** — per-symbol deep dive: metrics, prediction cards,
   candlestick chart with SMA 20/50, forecast comparison, price table,
   and the prediction-history audit trail.
4. **Paper Portfolio** — portfolio value trend, open/closed positions,
   exposure, and the BUY/SELL transaction log. Populated by
   `portfolio_engine.py`; empty states guide you if no data exists yet.

## Key Features

### 1. Indian Market Optimized
- NSE/BSE symbol format (e.g., `RELIANCE.NS`)
- INR currency throughout
- Indian market transaction costs (STT, brokerage, charges)
- Full Nifty 500 symbol list included (`nifty500_symbols.py`)

### 2. Ensemble ML Models
- **Three horizons**: daily (1d), weekly (5d), and monthly (20d) forecasts
- **LSTM**: Captures time-series patterns
- **XGBoost**: Handles tabular features
- **Dynamic weighting**: Models weighted by recent performance
- **Daily as a timing gate**: the entry engine uses the daily signal to
  delay otherwise-eligible entries (`WAIT_DAILY_TIMING`)

### 3. Comprehensive Features
- 40+ technical indicators (TA-Lib)
- Trend: SMA, EMA, MACD, ADX, Aroon
- Momentum: RSI, Stochastic, Williams %R, CCI
- Volatility: Bollinger Bands, ATR, Historical Volatility
- Volume: OBV, MFI, Volume Ratio

### 4. Risk Management
- Position sizing (fixed fractional + volatility-adjusted)
- Stop-loss and take-profit automation
- VaR and CVaR calculations
- Maximum drawdown monitoring
- Portfolio exposure limits

### 5. Realistic Backtesting
- Event-driven architecture
- Transaction costs (0.03% typical for Indian brokers)
- Slippage modeling (0.05%)
- Walk-forward validation
- Comprehensive metrics (Sharpe, Sortino, Max DD, Win Rate)

## Configuration

Edit `settings.py` or `.env` to customize:

```python
# Trading parameters
INITIAL_CAPITAL = 1000000  # INR
MAX_POSITION_SIZE = 0.10   # 10% per stock
TRANSACTION_COST = 0.0003  # 0.03%

# Risk management
MAX_DRAWDOWN = 0.15        # 15% max loss
STOP_LOSS_PCT = 0.05       # 5% stop loss
TAKE_PROFIT_PCT = 0.10     # 10% take profit

# Symbols to trade
NIFTY_500_SYMBOLS = get_nifty500_symbols()  # Full Nifty 500 universe
NIFTY_50_SYMBOLS = [...]                     # Smaller default watchlist
```

The PostgreSQL connection is configured via `POSTGRES_*` values in `.env`
(see `env.example`).

## Performance Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Total Return** | Overall profit/loss % | >10% annual |
| **Sharpe Ratio** | Risk-adjusted return | >1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | <15% |
| **Win Rate** | % of profitable trades | >55% |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Annualized Volatility** | Standard deviation of returns | 15-25% |

## Troubleshooting

### TA-Lib Installation Error
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Windows
# Download pre-compiled binary from:
# https://github.com/cgohlke/talib-binary
```

### No Data Fetched
- Check internet connection
- Verify symbol format (must include `.NS` for NSE)
- Yahoo Finance may have rate limits - wait and retry
- Ingesting all 500 symbols can trigger throttling. Use `--batch-size`
  and `--sleep` to slow things down, or `--limit` for a smaller run.

### PostgreSQL Connection Failed
- Confirm PostgreSQL is running and reachable
- Check the `POSTGRES_*` values in `.env` (see `env.example`)
- The ingestion and prediction scripts create the schema automatically
  on first connect

### Memory Error
- Reduce number of symbols
- Shorten data period (e.g., "1y" instead of "2y")
- Reduce LSTM lookback period

## Next Steps

1. **Refresh the Universe**: Update `_NIFTY_500_BASE` in `nifty500_symbols.py` when index membership changes
2. **Optimize Parameters**: Tune model hyperparameters in `ml_models.py`
3. **Add More Features**: Extend `feature_engineering.py` with custom indicators
4. **Schedule Ingestion**: Run `data_ingestion.py` on a schedule to keep prices current
5. **Connect Broker**: Integrate with Zerodha Kite API for live trading

## Pushing to GitHub

> **Note:** The included `.gitignore` only excludes the `logs/` folder,
> Python caches, and IDE/OS junk. Everything else — including `.env`,
> data files, models, and CSVs — is committed. The `.env` here only holds
> local `localhost` PostgreSQL defaults; if you later point it at a real
> remote database, add `.env` back to `.gitignore` and rotate any exposed
> credentials.

### One-time setup

1. Create an **empty** repository on GitHub (no README/.gitignore/license,
   so it doesn't conflict with these files). Copy its URL, e.g.
   `https://github.com/<you>/indian_stock_trader.git`.

2. Initialize the local repo and make the first commit:

   ```powershell
   git init
   git branch -M main
   git add .
   ```

3. Review what will be committed:

   ```powershell
   git status
   ```

4. Commit and push:

   ```powershell
   git commit -m "Initial commit: Indian Stock Trader (Nifty 500 pipeline)"
   git remote add origin https://github.com/<you>/indian_stock_trader.git
   git push -u origin main
   ```

   When prompted, authenticate with a **Personal Access Token** (GitHub
   no longer accepts account passwords over HTTPS). Create one at
   *GitHub → Settings → Developer settings → Personal access tokens*,
   with `repo` scope, and paste it as the password.

### Subsequent pushes

```powershell
git add .
git commit -m "Describe your change"
git push
```

### If you ever commit `.env` by accident

Rotate the exposed credentials immediately (they're considered
compromised once pushed), then remove the file from history — a plain
`git rm` in a new commit is not enough, since the secret stays in past
commits. Use a history-rewriting tool such as
[`git filter-repo`](https://github.com/newren/git-filter-repo) and force-push.

## Disclaimer

This system is for **educational and research purposes only**. Past performance does not guarantee future results. Always:
- Backtest thoroughly before live trading
- Start with small capital
- Use proper risk management
- Consult with a SEBI-registered investment advisor

## License

MIT License - Feel free to use and modify for your projects.

## Support

For issues or questions, check the code comments or add logging statements to debug.
