"""Entry point for market_analyzer CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--symbol", "-s", required=True, help="Stock ticker symbol (e.g., AAPL)")
@click.option("--analysis", "-a", type=click.Choice(["all", "fundamental", "technical", "valuation"]), default="all")
def main(symbol: str, analysis: str):
    """Analyze a stock with fundamentals, technicals and valuations."""
    console.print(f"[bold blue]Market Analyzer[/bold blue]")
    console.print(f"Analyzing: [green]{symbol.upper()}[/green]")
    console.print(f"Analysis type: {analysis}")

    # TODO: Implement analysis modules
    console.print("[yellow]Analysis modules coming soon...[/yellow]")


if __name__ == "__main__":
    main()
