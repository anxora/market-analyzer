"""
Backtesting module for the NVDA intraday trading strategy.

Tests the RSI + MACD + VWAP strategy against historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import yfinance as yf

from .indicators import TechnicalAnalyzer
from .signals import SignalGenerator, Signal, TradeSignal


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # "LONG" or "SHORT"
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "end_of_day"


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic stats
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L stats
    gross_profit: float
    gross_loss: float
    profit_factor: float  # gross_profit / gross_loss
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk stats
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Time stats
    avg_trade_duration: timedelta

    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def __str__(self) -> str:
        return f"""
════════════════════════════════════════════════════════════
                    BACKTEST RESULTS
════════════════════════════════════════════════════════════

RENDIMIENTO
  Capital Inicial:    ${self.initial_capital:,.2f}
  Capital Final:      ${self.final_capital:,.2f}
  Retorno Total:      ${self.total_return:,.2f} ({self.total_return_pct:+.2f}%)

TRADES
  Total Trades:       {self.total_trades}
  Ganadores:          {self.winning_trades} ({self.win_rate:.1f}%)
  Perdedores:         {self.losing_trades}

GANANCIAS/PÉRDIDAS
  Profit Bruto:       ${self.gross_profit:,.2f}
  Loss Bruto:         ${self.gross_loss:,.2f}
  Profit Factor:      {self.profit_factor:.2f}
  Promedio Win:       ${self.avg_win:,.2f}
  Promedio Loss:      ${self.avg_loss:,.2f}
  Mayor Ganancia:     ${self.largest_win:,.2f}
  Mayor Pérdida:      ${self.largest_loss:,.2f}

RIESGO
  Max Drawdown:       ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)
  Sharpe Ratio:       {self.sharpe_ratio:.2f}

TIEMPO
  Duración Promedio:  {self.avg_trade_duration}

════════════════════════════════════════════════════════════
"""


class Backtester:
    """
    Backtest the trading strategy against historical data.
    """

    def __init__(
        self,
        symbol: str = "NVDA",
        initial_capital: float = 100000,
        max_risk_pct: float = 0.5,
        commission_per_share: float = 0.005,  # $0.005 per share (IB rate)
        require_confluence: int = 2
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.max_risk_pct = max_risk_pct
        self.commission = commission_per_share

        self.signal_generator = SignalGenerator(
            require_confluence=require_confluence
        )

    def fetch_data(
        self,
        period: str = "1mo",
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch historical intraday data.

        Note: Yahoo Finance limits intraday data:
        - 5m interval: max 60 days
        - 1m interval: max 7 days
        """
        print(f"Descargando datos de {self.symbol} ({period}, {interval})...")

        data = yf.download(
            self.symbol,
            period=period,
            interval=interval,
            progress=False
        )

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        print(f"  {len(data)} barras descargadas")
        if len(data) > 0:
            print(f"  Desde: {data.index[0]}")
            print(f"  Hasta: {data.index[-1]}")

        return data

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        capital: float
    ) -> int:
        """Calculate position size based on risk."""
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            risk_per_share = entry_price * (self.max_risk_pct / 100)

        max_risk = capital * (self.max_risk_pct / 100)
        size = int(max_risk / risk_per_share)

        # Limit to what we can afford
        max_affordable = int(capital / entry_price)
        return min(size, max_affordable)

    def run(
        self,
        period: str = "1mo",
        interval: str = "5m",
        data: pd.DataFrame = None
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            period: Data period (e.g., "1mo", "3mo")
            interval: Bar interval (e.g., "5m", "15m")
            data: Optional pre-loaded data

        Returns:
            BacktestResult with all statistics
        """
        # Fetch data if not provided
        if data is None:
            data = self.fetch_data(period, interval)

        if len(data) < 50:
            raise ValueError("Insufficient data for backtesting")

        # Initialize
        capital = self.initial_capital
        position = None  # {"side", "entry_price", "quantity", "stop_loss", "take_profit", "entry_time"}
        trades: List[BacktestTrade] = []
        equity_curve = [capital]

        print(f"\nEjecutando backtest...")
        print(f"  Capital inicial: ${capital:,.2f}")
        print(f"  Riesgo máximo: {self.max_risk_pct}%")

        # Need enough bars for indicators
        min_bars = 30

        for i in range(min_bars, len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i]
            current_price = float(current_bar['Close'])
            current_high = float(current_bar['High'])
            current_low = float(current_bar['Low'])

            # Check if end of trading day (close positions)
            if hasattr(current_time, 'hour'):
                if current_time.hour >= 15 and current_time.minute >= 55:
                    if position is not None:
                        # Close position at end of day
                        pnl = self._close_position(
                            position, current_price, current_time, "end_of_day"
                        )
                        trades.append(pnl)
                        capital += pnl.pnl
                        position = None
                    equity_curve.append(capital)
                    continue

            # If we have a position, check for stop loss / take profit
            if position is not None:
                # Check stop loss
                if position['side'] == "LONG":
                    if current_low <= position['stop_loss']:
                        pnl = self._close_position(
                            position, position['stop_loss'], current_time, "stop_loss"
                        )
                        trades.append(pnl)
                        capital += pnl.pnl
                        position = None
                    elif current_high >= position['take_profit']:
                        pnl = self._close_position(
                            position, position['take_profit'], current_time, "take_profit"
                        )
                        trades.append(pnl)
                        capital += pnl.pnl
                        position = None
                else:  # SHORT
                    if current_high >= position['stop_loss']:
                        pnl = self._close_position(
                            position, position['stop_loss'], current_time, "stop_loss"
                        )
                        trades.append(pnl)
                        capital += pnl.pnl
                        position = None
                    elif current_low <= position['take_profit']:
                        pnl = self._close_position(
                            position, position['take_profit'], current_time, "take_profit"
                        )
                        trades.append(pnl)
                        capital += pnl.pnl
                        position = None

            # Generate signal using data up to current bar
            historical_data = data.iloc[:i+1]
            signal = self.signal_generator.generate_signal(historical_data)

            # Execute trades based on signal
            if position is None and signal.signal != Signal.HOLD:
                # Open new position
                if signal.signal == Signal.BUY:
                    stop_loss = current_price * (1 - self.max_risk_pct / 100)
                    take_profit = current_price * (1 + self.max_risk_pct * 2 / 100)
                    quantity = self.calculate_position_size(current_price, stop_loss, capital)

                    if quantity > 0:
                        commission = quantity * self.commission * 2  # Entry + exit
                        position = {
                            'side': "LONG",
                            'entry_price': current_price,
                            'quantity': quantity,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': current_time,
                            'commission': commission
                        }

                elif signal.signal == Signal.SELL and position is None:
                    # For now, only long positions (no shorting)
                    pass

            elif position is not None:
                # Check for exit signal (opposite signal)
                if position['side'] == "LONG" and signal.signal == Signal.SELL:
                    pnl = self._close_position(
                        position, current_price, current_time, "signal"
                    )
                    trades.append(pnl)
                    capital += pnl.pnl
                    position = None

            # Update equity curve
            if position is not None:
                # Mark to market
                unrealized = (current_price - position['entry_price']) * position['quantity']
                if position['side'] == "SHORT":
                    unrealized = -unrealized
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

        # Close any remaining position
        if position is not None:
            final_price = float(data['Close'].iloc[-1])
            pnl = self._close_position(
                position, final_price, data.index[-1], "end_of_backtest"
            )
            trades.append(pnl)
            capital += pnl.pnl

        # Calculate results
        return self._calculate_results(trades, equity_curve, capital)

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> BacktestTrade:
        """Close a position and return trade record."""
        if position['side'] == "LONG":
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']

        # Subtract commission
        pnl -= position.get('commission', 0)

        pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) * 100

        return BacktestTrade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            side=position['side'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )

    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        final_capital: float
    ) -> BacktestResult:
        """Calculate backtest statistics."""

        if not trades:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                gross_profit=0,
                gross_loss=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                avg_trade_duration=timedelta(0),
                trades=trades,
                equity_curve=pd.Series(equity_curve)
            )

        # Basic stats
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Trade stats
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        win_rate = (len(winning) / len(trades)) * 100 if trades else 0

        # P&L stats
        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = gross_loss / len(losing) if losing else 0

        largest_win = max(t.pnl for t in winning) if winning else 0
        largest_loss = min(t.pnl for t in losing) if losing else 0

        # Drawdown
        equity = pd.Series(equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = equity - rolling_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = (max_drawdown / rolling_max[drawdown.idxmin()]) * 100 if max_drawdown > 0 else 0

        # Sharpe ratio (simplified, using trade returns)
        if len(trades) > 1:
            returns = [t.pnl_pct for t in trades]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Average trade duration
        durations = [(t.exit_time - t.entry_time) for t in trades]
        avg_duration = sum(durations, timedelta(0)) / len(durations) if durations else timedelta(0)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            avg_trade_duration=avg_duration,
            trades=trades,
            equity_curve=equity
        )

    def print_trades(self, result: BacktestResult, limit: int = 20):
        """Print individual trades."""
        print(f"\nÚltimos {min(limit, len(result.trades))} trades:")
        print("-" * 80)

        for trade in result.trades[-limit:]:
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(
                f"{emoji} {trade.entry_time.strftime('%m/%d %H:%M')} → "
                f"{trade.exit_time.strftime('%H:%M')} | "
                f"{trade.side:5} {trade.quantity:4} @ ${trade.entry_price:.2f} → "
                f"${trade.exit_price:.2f} | "
                f"PnL: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%) | "
                f"{trade.exit_reason}"
            )


def run_backtest(
    symbol: str = "NVDA",
    period: str = "1mo",
    interval: str = "5m",
    capital: float = 100000,
    risk_pct: float = 0.5
) -> BacktestResult:
    """Convenience function to run a backtest."""
    backtester = Backtester(
        symbol=symbol,
        initial_capital=capital,
        max_risk_pct=risk_pct
    )

    result = backtester.run(period=period, interval=interval)
    print(result)
    backtester.print_trades(result)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backtest NVDA trading strategy')
    parser.add_argument('--symbol', default='NVDA', help='Stock symbol')
    parser.add_argument('--period', default='1mo', help='Data period (1mo, 3mo, etc)')
    parser.add_argument('--interval', default='5m', help='Bar interval (1m, 5m, 15m)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.5, help='Max risk per trade (%)')

    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        capital=args.capital,
        risk_pct=args.risk
    )
