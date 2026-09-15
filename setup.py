"""
Setup script for Indian Stock Trading System
Installs all required dependencies
"""
import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""

    requirements = [
        # Data handling
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'yfinance>=0.2.28',
        'ta-lib>=0.4.28',  # Technical analysis library

        # Machine Learning
        'scikit-learn>=1.3.0',
        'xgboost>=2.0.0',
        'tensorflow>=2.13.0',
        'joblib>=1.3.0',

        # Backtesting and visualization
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',

        # API and web
        'fastapi>=0.104.0',
        'uvicorn>=0.24.0',
        'pydantic>=2.0.0',

        # Utilities
        'python-dotenv>=1.0.0',
        'logging>=0.4.9.6',
        'tqdm>=4.66.0',

        # Monitoring
        'prometheus-client>=0.19.0',
    ]

    print("="*60)
    print("INDIAN STOCK TRADING SYSTEM - SETUP")
    print("="*60)
    print("\nInstalling required packages...\n")

    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            print(f"  Try manually: pip install {package}")

    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Install TA-Lib system library (if not already done):")
    print("   Ubuntu/Debian: sudo apt-get install ta-lib")
    print("   macOS: brew install ta-lib")
    print("   Windows: Download from https://github.com/cgohlke/talib-binary")
    print("\n2. Run the trading engine:")
    print("   python trading_engine.py")
    print("\n3. Or run the API server:")
    print("   python -m uvicorn api.main:app --reload")

if __name__ == "__main__":
    install_requirements()
