"""
Risk Management Module for Indian Stock Trading
Position sizing, stop-loss, volatility targeting, and portfolio constraints
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Comprehensive risk management for Indian equity trading
    """

    def __init__(self, 
                 initial_capital: float = 1000000,
                 max_position_pct: float = 0.10,
                 max_portfolio_exposure: float = 0.80,
                 max_drawdown_limit: float = 0.15,
                 volatility_target: float = 0.20,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 var_confidence: float = 0.95):
        """
        Initialize risk manager

        Args:
            initial_capital: Starting capital in INR
            max_position_pct: Max % of capital per position
            max_portfolio_exposure: Max % of capital invested
            max_drawdown_limit: Max allowed drawdown before stopping
            volatility_target: Target annualized volatility
            stop_loss_pct: Stop loss percentage per trade
            take_profit_pct: Take profit percentage per trade
            var_confidence: VaR confidence level
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_target = volatility_target
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.var_confidence = var_confidence

        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.positions = {}  # symbol -> {quantity, entry_price, stop_loss, take_profit}
        self.trade_history = []

    def calculate_position_size(self, symbol: str, price: float, 
                                signal_strength: float = 1.0,
                                volatility: Optional[float] = None) -> int:
        """
        Calculate optimal position size using multiple methods

        Args:
            symbol: Stock symbol
            price: Current price
            signal_strength: Confidence in signal (0-1)
            volatility: Stock's annualized volatility (optional)

        Returns:
            Number of shares to buy
        """
        # Method 1: Fixed fractional position sizing
        base_position_value = self.current_capital * self.max_position_pct

        # Adjust for signal strength
        adjusted_position = base_position_value * signal_strength

        # Method 2: Volatility-based position sizing (if volatility provided)
        if volatility:
            # Scale position inversely to volatility
            vol_adjustment = self.volatility_target / max(volatility, 0.01)
            adjusted_position *= vol_adjustment
            logger.debug(f"Volatility adjustment: {vol_adjustment:.2f}")

        # Calculate quantity
        quantity = int(adjusted_position / price)

        # Ensure minimum lot size (NSE typically 1 share for most stocks)
        if quantity < 1:
            quantity = 0
            logger.warning(f"Position size too small for {symbol}")

        # Check portfolio exposure
        current_exposure = self.get_portfolio_exposure(price)
        if current_exposure > self.max_portfolio_exposure:
            quantity = 0
            logger.warning(f"Max portfolio exposure reached")

        logger.info(f"Position size for {symbol}: {quantity} shares @ {price:.2f} INR")
        return quantity

    def get_portfolio_exposure(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio exposure as % of capital"""
        holdings_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return holdings_value / self.current_capital

    def set_stop_loss_take_profit(self, symbol: str, entry_price: float):
        """Set stop-loss and take-profit levels for a position"""
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)

        if symbol in self.positions:
            self.positions[symbol]['stop_loss'] = stop_loss
            self.positions[symbol]['take_profit'] = take_profit

        logger.info(f"{symbol}: Stop-loss={stop_loss:.2f}, Take-profit={take_profit:.2f}")
        return stop_loss, take_profit

    def check_stop_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if stop-loss or take-profit is triggered

        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', or None
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        stop_loss = pos.get('stop_loss', entry_price * (1 - self.stop_loss_pct))
        take_profit = pos.get('take_profit', entry_price * (1 + self.take_profit_pct))

        if current_price <= stop_loss:
            logger.warning(f"STOP LOSS triggered for {symbol} @ {current_price:.2f}")
            return 'STOP_LOSS'

        if current_price >= take_profit:
            logger.info(f"TAKE PROFIT triggered for {symbol} @ {current_price:.2f}")
            return 'TAKE_PROFIT'

        return None

    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Historical returns series
            confidence: Confidence level (default: self.var_confidence)

        Returns:
            VaR as positive percentage
        """
        confidence = confidence or self.var_confidence
        var = -returns.quantile(1 - confidence)
        return var * 100

    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)

        Args:
            returns: Historical returns series
            confidence: Confidence level

        Returns:
            CVaR as positive percentage
        """
        confidence = confidence or self.var_confidence
        var_threshold = returns.quantile(1 - confidence)
        cvar = -returns[returns <= var_threshold].mean()
        return cvar * 100 if not np.isnan(cvar) else 0

    def update_capital(self, new_capital: float):
        """Update current capital and track drawdown"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if current_drawdown > self.max_drawdown_limit:
            logger.error(f"MAX DRAWDOWN EXCEEDED: {current_drawdown:.2%}")
            return False  # Trading should stop

        return True

    def get_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics"""
        if returns.empty or len(returns) < 10:
            return {}

        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualized (assuming daily returns)
        ann_return = mean_return * 252
        ann_vol = std_return * np.sqrt(252)

        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_std if downside_std > 0 else 0

        # VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        metrics = {
            'annualized_return': round(ann_return * 100, 2),
            'annualized_volatility': round(ann_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'var_95_pct': round(var_95, 2),
            'var_99_pct': round(var_99, 2),
            'cvar_95_pct': round(cvar_95, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'skewness': round(skew, 2),
            'kurtosis': round(kurt, 2),
            'daily_win_rate': round((returns > 0).mean() * 100, 2)
        }

        return metrics

    def add_position(self, symbol: str, quantity: int, entry_price: float):
        """Add new position to portfolio"""
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'take_profit': entry_price * (1 + self.take_profit_pct),
            'date_added': pd.Timestamp.now()
        }
        logger.info(f"Added position: {quantity} {symbol} @ {entry_price:.2f}")

    def remove_position(self, symbol: str, exit_price: float):
        """Remove position from portfolio and record P&L"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pnl = (exit_price - pos['entry_price']) * pos['quantity']
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

        self.trade_history.append({
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'date_closed': pd.Timestamp.now()
        })

        del self.positions[symbol]
        logger.info(f"Closed position: {symbol}, P&L: {pnl:.2f} INR ({pnl_pct:.2f}%)")


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    rm = RiskManager(initial_capital=1000000)

    # Calculate position size
    quantity = rm.calculate_position_size("RELIANCE.NS", price=2500, signal_strength=0.8)
    print(f"Position size: {quantity} shares")

    # Add position
    if quantity > 0:
        rm.add_position("RELIANCE.NS", quantity, entry_price=2500)

        # Set stop-loss and take-profit
        rm.set_stop_loss_take_profit("RELIANCE.NS", 2500)

        # Check stop conditions
        action = rm.check_stop_conditions("RELIANCE.NS", 2375)  # Stop loss hit
        print(f"Action: {action}")

    # Generate sample returns for risk metrics
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02 + 0.0005)

    # Calculate risk metrics
    metrics = rm.get_risk_metrics(returns)

    print("\n" + "="*50)
    print("RISK METRICS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
