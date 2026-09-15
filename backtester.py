"""
Backtesting Engine for Indian Stock Market
Walk-forward validation, realistic transaction costs, and risk management
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade record"""
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    position: int = 0  # 1=long, -1=short
    quantity: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0

@dataclass
class PortfolioState:
    """Portfolio state at any point in time"""
    date: str
    cash: float
    holdings: Dict[str, float]
    total_value: float

class Backtester:
    """
    Event-driven backtesting engine for Indian markets
    """

    def __init__(self, initial_capital: float = 1000000,
                 transaction_cost: float = 0.0003,
                 slippage: float = 0.0005,
                 max_position_pct: float = 0.10):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital in INR
            transaction_cost: Total transaction cost (brokerage + STT + charges)
            slippage: Assumed slippage per trade
            max_position_pct: Maximum position size as % of portfolio
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_pct = max_position_pct

        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.holdings = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_values = []
        self.current_date = None

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        holdings_value = sum(
            qty * prices.get(symbol, 0) 
            for symbol, qty in self.holdings.items()
        )
        return self.cash + holdings_value

    def buy(self, symbol: str, price: float, quantity: int, date: str):
        """Execute buy order"""
        # Calculate cost with slippage and transaction cost
        effective_price = price * (1 + self.slippage)
        cost = effective_price * quantity * (1 + self.transaction_cost)

        if cost > self.cash:
            # Buy as much as possible
            max_qty = int(self.cash / (effective_price * (1 + self.transaction_cost)))
            if max_qty == 0:
                logger.warning(f"Insufficient cash to buy {symbol}")
                return
            quantity = max_qty
            cost = effective_price * quantity * (1 + self.transaction_cost)

        # Update portfolio
        self.cash -= cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity

        # Record trade
        trade = Trade(
            entry_date=date,
            entry_price=effective_price,
            position=1,
            quantity=quantity
        )
        self.trades.append(trade)

        logger.debug(f"BUY {quantity} {symbol} @ {effective_price:.2f} INR")

    def sell(self, symbol: str, price: float, quantity: int, date: str):
        """Execute sell order"""
        if symbol not in self.holdings or self.holdings[symbol] == 0:
            logger.warning(f"No holdings to sell for {symbol}")
            return

        # Sell only what we have
        quantity = min(quantity, self.holdings[symbol])

        # Calculate proceeds with slippage and transaction cost
        effective_price = price * (1 - self.slippage)
        proceeds = effective_price * quantity * (1 - self.transaction_cost)

        # Update portfolio
        self.cash += proceeds
        self.holdings[symbol] -= quantity

        if self.holdings[symbol] == 0:
            del self.holdings[symbol]

        # Update last trade for this symbol
        for trade in reversed(self.trades):
            if trade.entry_date and trade.position == 1:
                # Find matching buy trade (simplified - assumes LIFO)
                trade.exit_date = date
                trade.exit_price = effective_price
                trade.pnl = (effective_price - trade.entry_price) * trade.quantity
                trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
                break

        logger.debug(f"SELL {quantity} {symbol} @ {effective_price:.2f} INR")

    def run_backtest(self, df: pd.DataFrame, signals: pd.Series,
                     symbol: str = "TEST") -> pd.DataFrame:
        """
        Run backtest on price data with trading signals

        Args:
            df: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            symbol: Stock symbol

        Returns:
            DataFrame with daily portfolio values
        """
        self.reset()

        prices = {}
        daily_returns = []

        for idx, row in df.iterrows():
            date = str(row.get('date', idx))
            price = row['close']
            signal = signals.iloc[idx] if hasattr(signals, 'iloc') else signals[idx]

            prices[symbol] = price

            # Execute trades based on signals
            if signal > 0.5:  # Buy signal
                # Calculate position size
                position_value = self.get_portfolio_value(prices) * self.max_position_pct
                quantity = int(position_value / price)

                if quantity > 0:
                    self.buy(symbol, price, quantity, date)

            elif signal < -0.5:  # Sell signal
                quantity = self.holdings.get(symbol, 0)
                if quantity > 0:
                    self.sell(symbol, price, quantity, date)

            # Record portfolio value
            portfolio_value = self.get_portfolio_value(prices)
            self.portfolio_values.append({
                'date': date,
                'cash': self.cash,
                'holdings_value': portfolio_value - self.cash,
                'total_value': portfolio_value,
                'return': (portfolio_value / self.initial_capital - 1) * 100
            })

        return pd.DataFrame(self.portfolio_values)

    def calculate_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        if portfolio_df.empty:
            return {}

        # Daily returns
        daily_returns = portfolio_df['total_value'].pct_change().dropna()

        # Total return
        total_return = (portfolio_df['total_value'].iloc[-1] / self.initial_capital - 1) * 100

        # Annualized return (assuming 252 trading days)
        n_days = len(portfolio_df)
        annualized_return = ((1 + total_return/100) ** (252/n_days) - 1) * 100

        # Volatility
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (annualized_return / 100) / (annual_vol / 100) if annual_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Number of trades
        n_trades = len(self.trades)

        metrics = {
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': round(annualized_return, 2),
            'annual_volatility_pct': round(annual_vol, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf',
            'total_trades': n_trades,
            'final_value_inr': round(portfolio_df['total_value'].iloc[-1], 2)
        }

        return metrics

    def plot_results(self, portfolio_df: pd.DataFrame, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Portfolio value
        axes[0].plot(portfolio_df['date'], portfolio_df['total_value'])
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('Portfolio Value (INR)')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Returns
        axes[1].plot(portfolio_df['date'], portfolio_df['return'])
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Cumulative Return (%)')
        axes[1].set_ylabel('Return %')
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        daily_returns = portfolio_df['total_value'].pct_change()
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        axes[2].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        axes[2].plot(range(len(drawdown)), drawdown, color='red')
        axes[2].set_title('Drawdown (%)')
        axes[2].set_ylabel('Drawdown %')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample price data
    np.random.seed(42)
    n = 252  # 1 year of trading days
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Generate random walk with drift
    returns = np.random.randn(n) * 0.02 + 0.0005
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })

    # Generate signals (simple moving average crossover)
    sma_10 = df['close'].rolling(10).mean()
    sma_20 = df['close'].rolling(20).mean()
    signals = np.where(sma_10 > sma_20, 1, -1)

    # Run backtest
    backtester = Backtester(initial_capital=1000000)
    results = backtester.run_backtest(df, pd.Series(signals))

    # Calculate metrics
    metrics = backtester.calculate_metrics(results)

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Plot results
    backtester.plot_results(results, save_path="backtest_results.png")
