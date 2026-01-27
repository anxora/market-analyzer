"""Entry point for market_analyzer CLI."""

import click
from rich.console import Console
from rich.table import Table

from .data import fetch_historical_data, fetch_yesterday_data
from .backtest import BacktestEngine

console = Console()


@click.group()
def cli():
    """Market Analyzer - Stock analysis and backtesting tool."""
    pass


@cli.command()
@click.option("--symbol", "-s", required=True, help="Stock ticker symbol (e.g., AAPL)")
@click.option("--analysis", "-a", type=click.Choice(["all", "fundamental", "technical", "valuation"]), default="all")
def analyze(symbol: str, analysis: str):
    """Analyze a stock with fundamentals, technicals and valuations."""
    console.print(f"[bold blue]Market Analyzer[/bold blue]")
    console.print(f"Analyzing: [green]{symbol.upper()}[/green]")
    console.print(f"Analysis type: {analysis}")

    # TODO: Implement analysis modules
    console.print("[yellow]Analysis modules coming soon...[/yellow]")


@cli.command()
@click.option("--symbol", "-s", required=True, help="Stock ticker symbol (e.g., AAPL)")
@click.option("--strategy", "-st", type=click.Choice(["sma", "rsi", "momentum"]), default="sma",
              help="Trading strategy to backtest")
@click.option("--capital", "-c", default=10000.0, help="Initial capital for backtest")
@click.option("--period", "-p", default="1mo", help="Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y)")
@click.option("--yesterday", "-y", is_flag=True, help="Run backtest on yesterday's data only")
def backtest(symbol: str, strategy: str, capital: float, period: str, yesterday: bool):
    """Run backtest on a trading strategy."""
    console.print(f"[bold blue]Market Analyzer - Backtest[/bold blue]")
    console.print(f"Symbol: [green]{symbol.upper()}[/green]")
    console.print(f"Strategy: [cyan]{strategy.upper()}[/cyan]")
    console.print(f"Initial Capital: [yellow]${capital:,.2f}[/yellow]")
    console.print()

    with console.status("[bold green]Fetching historical data..."):
        if yesterday:
            df = fetch_yesterday_data(symbol)
        else:
            df = fetch_historical_data(symbol, period=period)

    if df.empty:
        console.print("[red]No data available for the specified symbol/period.[/red]")
        return

    console.print(f"[dim]Loaded {len(df)} data points[/dim]")
    console.print()

    with console.status("[bold green]Running backtest..."):
        engine = BacktestEngine(initial_capital=capital)
        result = engine.run(df, symbol.upper(), strategy)

    # Display results
    console.print(result.summary())

    # Show trades table if there are any
    if result.trades:
        console.print()
        table = Table(title="Trade History")
        table.add_column("Entry Date", style="cyan")
        table.add_column("Entry Price", justify="right")
        table.add_column("Exit Date", style="cyan")
        table.add_column("Exit Price", justify="right")
        table.add_column("Shares", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("P/L %", justify="right")

        for trade in result.trades:
            pl_color = "green" if trade.profit_loss >= 0 else "red"
            table.add_row(
                trade.entry_time.strftime("%Y-%m-%d %H:%M"),
                f"${trade.entry_price:.2f}",
                trade.exit_time.strftime("%Y-%m-%d %H:%M"),
                f"${trade.exit_price:.2f}",
                str(trade.shares),
                f"[{pl_color}]${trade.profit_loss:+,.2f}[/{pl_color}]",
                f"[{pl_color}]{trade.profit_loss_pct:+.2f}%[/{pl_color}]",
            )

        console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
