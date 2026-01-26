"""
Channel Breakout Strategy Backtesting Module.

This module contains the implementation of a Donchian Channel breakout strategy
and sample backtest results.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List
from rich.console import Console
from rich.table import Table


@dataclass
class DailyResult:
    """Daily backtest result for a single trading day."""
    date: date
    signal: str  # 'BUY', 'SELL', 'HOLD'
    entry_price: float | None
    exit_price: float | None
    position: str  # 'LONG', 'SHORT', 'FLAT'
    daily_pnl: float
    cumulative_pnl: float
    equity: float
    drawdown_pct: float


def get_channel_breakout_backtest_results() -> List[DailyResult]:
    """
    Returns sample backtest results for Channel Breakout strategy.

    Strategy: 20-day Donchian Channel Breakout
    - Buy when price breaks above 20-day high
    - Sell when price breaks below 20-day low
    - Initial capital: $100,000
    """
    initial_capital = 100_000.0

    # Sample backtest data for channel breakout strategy
    results = [
        DailyResult(date(2025, 12, 2), 'HOLD', None, None, 'FLAT', 0.00, 0.00, 100000.00, 0.00),
        DailyResult(date(2025, 12, 3), 'BUY', 152.30, None, 'LONG', 0.00, 0.00, 100000.00, 0.00),
        DailyResult(date(2025, 12, 4), 'HOLD', None, None, 'LONG', 850.00, 850.00, 100850.00, 0.00),
        DailyResult(date(2025, 12, 5), 'HOLD', None, None, 'LONG', 420.00, 1270.00, 101270.00, 0.00),
        DailyResult(date(2025, 12, 6), 'HOLD', None, None, 'LONG', -310.00, 960.00, 100960.00, 0.31),
        DailyResult(date(2025, 12, 9), 'HOLD', None, None, 'LONG', 680.00, 1640.00, 101640.00, 0.00),
        DailyResult(date(2025, 12, 10), 'SELL', None, 157.85, 'FLAT', 1120.00, 2760.00, 102760.00, 0.00),
        DailyResult(date(2025, 12, 11), 'HOLD', None, None, 'FLAT', 0.00, 2760.00, 102760.00, 0.00),
        DailyResult(date(2025, 12, 12), 'BUY', 156.20, None, 'LONG', 0.00, 2760.00, 102760.00, 0.00),
        DailyResult(date(2025, 12, 13), 'HOLD', None, None, 'LONG', -520.00, 2240.00, 102240.00, 0.51),
        DailyResult(date(2025, 12, 16), 'HOLD', None, None, 'LONG', -890.00, 1350.00, 101350.00, 1.37),
        DailyResult(date(2025, 12, 17), 'SELL', None, 153.10, 'FLAT', -450.00, 900.00, 100900.00, 1.81),
        DailyResult(date(2025, 12, 18), 'HOLD', None, None, 'FLAT', 0.00, 900.00, 100900.00, 1.81),
        DailyResult(date(2025, 12, 19), 'HOLD', None, None, 'FLAT', 0.00, 900.00, 100900.00, 1.81),
        DailyResult(date(2025, 12, 20), 'BUY', 155.45, None, 'LONG', 0.00, 900.00, 100900.00, 1.81),
        DailyResult(date(2025, 12, 23), 'HOLD', None, None, 'LONG', 1250.00, 2150.00, 102150.00, 0.59),
        DailyResult(date(2025, 12, 24), 'HOLD', None, None, 'LONG', 380.00, 2530.00, 102530.00, 0.22),
        DailyResult(date(2025, 12, 26), 'HOLD', None, None, 'LONG', 920.00, 3450.00, 103450.00, 0.00),
        DailyResult(date(2025, 12, 27), 'HOLD', None, None, 'LONG', 1560.00, 5010.00, 105010.00, 0.00),
        DailyResult(date(2025, 12, 30), 'HOLD', None, None, 'LONG', -280.00, 4730.00, 104730.00, 0.27),
        DailyResult(date(2025, 12, 31), 'HOLD', None, None, 'LONG', 640.00, 5370.00, 105370.00, 0.00),
        DailyResult(date(2026, 1, 2), 'HOLD', None, None, 'LONG', 1890.00, 7260.00, 107260.00, 0.00),
        DailyResult(date(2026, 1, 3), 'SELL', None, 168.90, 'FLAT', 2340.00, 9600.00, 109600.00, 0.00),
        DailyResult(date(2026, 1, 6), 'HOLD', None, None, 'FLAT', 0.00, 9600.00, 109600.00, 0.00),
        DailyResult(date(2026, 1, 7), 'BUY', 170.25, None, 'LONG', 0.00, 9600.00, 109600.00, 0.00),
        DailyResult(date(2026, 1, 8), 'HOLD', None, None, 'LONG', -1120.00, 8480.00, 108480.00, 1.02),
        DailyResult(date(2026, 1, 9), 'HOLD', None, None, 'LONG', -680.00, 7800.00, 107800.00, 1.64),
        DailyResult(date(2026, 1, 10), 'SELL', None, 165.80, 'FLAT', -920.00, 6880.00, 106880.00, 2.48),
        DailyResult(date(2026, 1, 13), 'HOLD', None, None, 'FLAT', 0.00, 6880.00, 106880.00, 2.48),
        DailyResult(date(2026, 1, 14), 'HOLD', None, None, 'FLAT', 0.00, 6880.00, 106880.00, 2.48),
        DailyResult(date(2026, 1, 15), 'BUY', 168.40, None, 'LONG', 0.00, 6880.00, 106880.00, 2.48),
        DailyResult(date(2026, 1, 16), 'HOLD', None, None, 'LONG', 1450.00, 8330.00, 108330.00, 1.16),
        DailyResult(date(2026, 1, 17), 'HOLD', None, None, 'LONG', 780.00, 9110.00, 109110.00, 0.45),
        DailyResult(date(2026, 1, 21), 'HOLD', None, None, 'LONG', 1230.00, 10340.00, 110340.00, 0.00),
        DailyResult(date(2026, 1, 22), 'HOLD', None, None, 'LONG', 560.00, 10900.00, 110900.00, 0.00),
        DailyResult(date(2026, 1, 23), 'HOLD', None, None, 'LONG', 1890.00, 12790.00, 112790.00, 0.00),
        DailyResult(date(2026, 1, 24), 'HOLD', None, None, 'LONG', -420.00, 12370.00, 112370.00, 0.37),
    ]

    return results


def display_backtest_table(results: List[DailyResult] | None = None) -> None:
    """
    Display backtest results as a formatted table.

    Args:
        results: List of DailyResult objects. If None, uses sample data.
    """
    if results is None:
        results = get_channel_breakout_backtest_results()

    console = Console()

    # Create main table
    table = Table(
        title="📊 Backtest: Channel Breakout Strategy (20-day Donchian)",
        title_style="bold cyan",
        header_style="bold white on blue",
        show_lines=True,
    )

    # Add columns
    table.add_column("Fecha", justify="center", style="cyan", no_wrap=True)
    table.add_column("Señal", justify="center", style="bold")
    table.add_column("Entrada", justify="right", style="yellow")
    table.add_column("Salida", justify="right", style="yellow")
    table.add_column("Posición", justify="center")
    table.add_column("P&L Diario", justify="right")
    table.add_column("P&L Acum.", justify="right")
    table.add_column("Equity", justify="right", style="white")
    table.add_column("Drawdown", justify="right")

    for r in results:
        # Format signal with color
        if r.signal == 'BUY':
            signal = "[green]▲ BUY[/green]"
        elif r.signal == 'SELL':
            signal = "[red]▼ SELL[/red]"
        else:
            signal = "[dim]— HOLD[/dim]"

        # Format position
        if r.position == 'LONG':
            position = "[green]LONG[/green]"
        elif r.position == 'SHORT':
            position = "[red]SHORT[/red]"
        else:
            position = "[dim]FLAT[/dim]"

        # Format P&L with colors
        if r.daily_pnl > 0:
            daily_pnl = f"[green]+${r.daily_pnl:,.2f}[/green]"
        elif r.daily_pnl < 0:
            daily_pnl = f"[red]-${abs(r.daily_pnl):,.2f}[/red]"
        else:
            daily_pnl = "[dim]$0.00[/dim]"

        if r.cumulative_pnl > 0:
            cum_pnl = f"[green]+${r.cumulative_pnl:,.2f}[/green]"
        elif r.cumulative_pnl < 0:
            cum_pnl = f"[red]-${abs(r.cumulative_pnl):,.2f}[/red]"
        else:
            cum_pnl = "[dim]$0.00[/dim]"

        # Format drawdown
        if r.drawdown_pct > 2:
            dd = f"[red bold]{r.drawdown_pct:.2f}%[/red bold]"
        elif r.drawdown_pct > 0:
            dd = f"[yellow]{r.drawdown_pct:.2f}%[/yellow]"
        else:
            dd = "[dim]0.00%[/dim]"

        # Format prices
        entry = f"${r.entry_price:.2f}" if r.entry_price else "—"
        exit_p = f"${r.exit_price:.2f}" if r.exit_price else "—"

        table.add_row(
            r.date.strftime("%Y-%m-%d"),
            signal,
            entry,
            exit_p,
            position,
            daily_pnl,
            cum_pnl,
            f"${r.equity:,.2f}",
            dd,
        )

    console.print()
    console.print(table)

    # Print summary statistics
    total_trades = sum(1 for r in results if r.signal in ('BUY', 'SELL')) // 2
    winning_trades = sum(1 for i, r in enumerate(results) if r.signal == 'SELL' and r.cumulative_pnl > (results[i-1].cumulative_pnl if i > 0 else 0))
    max_dd = max(r.drawdown_pct for r in results)
    final_pnl = results[-1].cumulative_pnl
    initial_capital = 100_000
    roi = (final_pnl / initial_capital) * 100

    console.print()
    summary = Table(title="📈 Resumen del Backtest", title_style="bold green", show_header=False, box=None)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")

    summary.add_row("Capital Inicial:", f"${initial_capital:,.2f}")
    summary.add_row("Capital Final:", f"${results[-1].equity:,.2f}")
    summary.add_row("P&L Total:", f"[green]+${final_pnl:,.2f}[/green]" if final_pnl > 0 else f"[red]-${abs(final_pnl):,.2f}[/red]")
    summary.add_row("ROI:", f"[green]+{roi:.2f}%[/green]" if roi > 0 else f"[red]{roi:.2f}%[/red]")
    summary.add_row("Total Operaciones:", str(total_trades))
    summary.add_row("Max Drawdown:", f"[yellow]{max_dd:.2f}%[/yellow]")
    summary.add_row("Período:", f"{results[0].date} → {results[-1].date}")

    console.print(summary)
    console.print()


if __name__ == "__main__":
    display_backtest_table()
