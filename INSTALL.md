# Installation and Quick Start Guide

## Step 1: Install System Dependencies

### Ubuntu/Debian (Linux)
```bash
sudo apt-get update
sudo apt-get install -y ta-lib
sudo apt-get install -y python3 python3-pip python3-venv
```

### macOS
```bash
brew install ta-lib
brew install python
```

### Windows
Download TA-Lib binary from:
https://github.com/cgohlke/talib-binary/releases

## Step 2: Set Up Python Environment

```bash
cd indian_stock_trader
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the System

```bash
# Quick example
python example_runner.py

# Full pipeline
python trading_engine.py
```

## Troubleshooting

### Error: "No module named 'talib'"
Install TA-Lib system library (Step 1)

### Error: "No data fetched"
- Check internet connection
- Verify symbol format (e.g., "RELIANCE.NS")

### Error: "Memory Error"
- Reduce number of symbols
- Shorten data period

## Full Documentation

See README.md for complete documentation.
