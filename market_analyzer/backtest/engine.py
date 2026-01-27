"""Backtesting engine for volatility breakout strategy."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import pandas as pd

from .strategies import Signal, VolatilityBreakoutStrategy


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
            f"  BACKTEST: {self.symbol} - VOLATILITY BREAKOUT",
            f"═══════════════════════════════════════════════════",
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
    """Engine for running volatility breakout backtests."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        k: float = 0.5,
        commission: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.k = k
        self.commission = commission
        self.strategy = VolatilityBreakoutStrategy(k=k)

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Run volatility breakout backtest.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock ticker symbol

        Returns:
            BacktestResult with performance metrics
        """
        df_signals = self.strategy.calculate_signals(df)
        return self._simulate_trades(df_signals, symbol)

    def _simulate_trades(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Simulate volatility breakout trades.

        For each day with a BUY signal:
        - Enter at the breakout level
        - Exit at the close of the same day
        """
        capital = self.initial_capital
        trades: List[Trade] = []

        for idx, row in df.iterrows():
            if row["Signal"] == Signal.BUY.value and pd.notna(row["Breakout_Level"]):
                # Entry at breakout level
                entry_price = row["Breakout_Level"]
                exit_price = row["Close"]

                # Calculate shares
                commission_cost = capital * self.commission
                available = capital - commission_cost
                shares = int(available / entry_price)

                if shares > 0:
                    # Calculate P/L
                    cost = shares * entry_price + commission_cost
                    sale = shares * exit_price
                    sale_commission = sale * self.commission
                    net_sale = sale - sale_commission

                    profit_loss = net_sale - cost
                    profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100

                    # Update capital
                    capital = capital + profit_loss

                    trades.append(Trade(
                        entry_time=idx,
                        entry_price=entry_price,
                        exit_time=idx,
                        exit_price=exit_price,
                        shares=shares,
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
