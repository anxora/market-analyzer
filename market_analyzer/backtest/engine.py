"""Backtesting engine for trading strategies."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .strategies import Signal, SMAStrategy, RSIStrategy, MomentumStrategy, BreakoutStrategy


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    shares: int
    profit_loss: float
    profit_loss_pct: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    trades: List[Trade] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary string of backtest results."""
        lines = [
            f"═══════════════════════════════════════════════════",
            f"  BACKTEST RESULTS: {self.symbol}",
            f"═══════════════════════════════════════════════════",
            f"  Strategy:         {self.strategy_name}",
            f"  Period:           {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            f"───────────────────────────────────────────────────",
            f"  Initial Capital:  ${self.initial_capital:,.2f}",
            f"  Final Capital:    ${self.final_capital:,.2f}",
            f"  Total Return:     ${self.total_return:,.2f} ({self.total_return_pct:+.2f}%)",
            f"───────────────────────────────────────────────────",
            f"  Total Trades:     {self.num_trades}",
            f"  Winning Trades:   {self.winning_trades}",
            f"  Losing Trades:    {self.losing_trades}",
            f"  Win Rate:         {self.win_rate:.1f}%",
            f"═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Engine for running backtests on trading strategies."""

    STRATEGIES = {
        "breakout": BreakoutStrategy,
        "sma": SMAStrategy,
        "rsi": RSIStrategy,
        "momentum": MomentumStrategy,
    }

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission per trade
    ):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_name: str = "sma",
        **strategy_kwargs
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock ticker symbol
            strategy_name: Name of strategy to use ('sma', 'rsi', 'momentum')
            **strategy_kwargs: Additional arguments for the strategy

        Returns:
            BacktestResult with performance metrics
        """
        if strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = self.STRATEGIES[strategy_name](**strategy_kwargs)
        df_signals = strategy.calculate_signals(df)

        return self._simulate_trades(df_signals, symbol, strategy_name)

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_name: str
    ) -> BacktestResult:
        """Simulate trades based on signals.

        Args:
            df: DataFrame with signals
            symbol: Stock ticker
            strategy_name: Name of the strategy used

        Returns:
            BacktestResult with all metrics
        """
        capital = self.initial_capital
        position = 0  # Number of shares held
        entry_price = 0.0
        entry_time = None
        trades: List[Trade] = []

        for idx, row in df.iterrows():
            price = row["Close"]
            signal = row["Signal"]

            # Buy signal and not in position
            if signal == Signal.BUY.value and position == 0:
                # Calculate shares to buy (use all capital)
                commission_cost = capital * self.commission
                available = capital - commission_cost
                position = int(available / price)

                if position > 0:
                    entry_price = price
                    entry_time = idx
                    capital -= (position * price) + commission_cost

            # Sell signal and in position
            elif signal == Signal.SELL.value and position > 0:
                # Sell all shares
                sale_value = position * price
                commission_cost = sale_value * self.commission
                capital += sale_value - commission_cost

                # Record trade
                profit_loss = (price - entry_price) * position - (2 * commission_cost)
                profit_loss_pct = ((price - entry_price) / entry_price) * 100

                trades.append(Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=idx,
                    exit_price=price,
                    shares=position,
                    profit_loss=profit_loss,
                    profit_loss_pct=profit_loss_pct,
                ))

                position = 0
                entry_price = 0.0
                entry_time = None

        # Close any remaining position at the last price
        if position > 0:
            last_price = df["Close"].iloc[-1]
            last_time = df.index[-1]
            sale_value = position * last_price
            commission_cost = sale_value * self.commission
            capital += sale_value - commission_cost

            profit_loss = (last_price - entry_price) * position - (2 * commission_cost)
            profit_loss_pct = ((last_price - entry_price) / entry_price) * 100

            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=last_time,
                exit_price=last_price,
                shares=position,
                profit_loss=profit_loss,
                profit_loss_pct=profit_loss_pct,
            ))

        # Calculate metrics
        winning_trades = sum(1 for t in trades if t.profit_loss > 0)
        losing_trades = sum(1 for t in trades if t.profit_loss <= 0)
        win_rate = (winning_trades / len(trades) * 100) if trades else 0.0

        total_return = capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name.upper(),
            start_date=df.index[0].to_pydatetime(),
            end_date=df.index[-1].to_pydatetime(),
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            trades=trades,
        )
