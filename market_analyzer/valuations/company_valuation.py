"""
Company Valuation Analysis

Analyzes the valuation (Market Cap vs Net Income) of the world's largest companies
using trailing twelve months (TTM) net income data.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import get_valuation_data


# Top 25 largest companies by market cap (as of 2024-2025)
TOP_COMPANIES = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'NVDA',   # NVIDIA
    'GOOGL',  # Alphabet (Google)
    'AMZN',   # Amazon
    'META',   # Meta Platforms
    'BRK-B',  # Berkshire Hathaway
    'TSM',    # Taiwan Semiconductor
    'LLY',    # Eli Lilly
    'AVGO',   # Broadcom
    'JPM',    # JPMorgan Chase
    'TSLA',   # Tesla
    'V',      # Visa
    'WMT',    # Walmart
    'XOM',    # ExxonMobil
    'UNH',    # UnitedHealth
    'MA',     # Mastercard
    'NVO',    # Novo Nordisk
    'JNJ',    # Johnson & Johnson
    'PG',     # Procter & Gamble
    'HD',     # Home Depot
    'COST',   # Costco
    'ORCL',   # Oracle
    'ABBV',   # AbbVie
    'BAC',    # Bank of America
]


def format_currency(value: float, in_billions: bool = True) -> str:
    """Format currency value for display."""
    if value is None:
        return "N/A"
    if in_billions:
        return f"${value / 1e9:,.1f}B"
    return f"${value:,.0f}"


def format_pe(value: float) -> str:
    """Format P/E ratio for display."""
    if value is None:
        return "N/A"
    if value < 0:
        return "Negative"
    return f"{value:.1f}x"


def analyze_top_companies(tickers: List[str] = None) -> pd.DataFrame:
    """
    Analyze valuation metrics for a list of companies.

    Args:
        tickers: List of stock tickers. Defaults to TOP_COMPANIES.

    Returns:
        DataFrame with valuation analysis
    """
    if tickers is None:
        tickers = TOP_COMPANIES

    console = Console()
    console.print("\n[bold cyan]Fetching valuation data for top companies...[/bold cyan]\n")

    results = []
    for i, ticker in enumerate(tickers, 1):
        console.print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end="\r")
        data = get_valuation_data(ticker)
        results.append(data)

    console.print(" " * 50)  # Clear the line

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by market cap descending
    df = df.sort_values('market_cap', ascending=False, na_position='last')

    # Reset index
    df = df.reset_index(drop=True)

    return df


def display_valuation_table(df: pd.DataFrame) -> None:
    """Display valuation analysis as a rich table."""
    console = Console()

    # Create the main table
    table = Table(
        title="Company Valuation Analysis - Market Cap vs TTM Net Income",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Company", style="white", width=25)
    table.add_column("Sector", style="dim", width=18)
    table.add_column("Market Cap", justify="right", style="green", width=12)
    table.add_column("TTM Net Income", justify="right", style="yellow", width=14)
    table.add_column("P/E Ratio", justify="right", style="magenta", width=10)

    for idx, row in df.iterrows():
        # Color P/E based on value
        pe_val = row['pe_ratio']
        if pe_val is None:
            pe_style = "dim"
        elif pe_val < 0:
            pe_style = "red"
        elif pe_val < 15:
            pe_style = "green"
        elif pe_val < 25:
            pe_style = "yellow"
        elif pe_val < 40:
            pe_style = "orange1"
        else:
            pe_style = "red"

        table.add_row(
            str(idx + 1),
            row['ticker'],
            row['name'][:25] if row['name'] else 'N/A',
            row['sector'][:18] if row['sector'] else 'N/A',
            format_currency(row['market_cap']),
            format_currency(row['ttm_net_income']),
            f"[{pe_style}]{format_pe(row['pe_ratio'])}[/{pe_style}]",
        )

    console.print(table)


def display_summary_stats(df: pd.DataFrame) -> None:
    """Display summary statistics."""
    console = Console()

    # Filter valid data
    valid_pe = df[df['pe_ratio'].notna() & (df['pe_ratio'] > 0)]
    valid_income = df[df['ttm_net_income'].notna() & (df['ttm_net_income'] > 0)]

    # Calculate stats
    total_market_cap = df['market_cap'].sum()
    total_net_income = valid_income['ttm_net_income'].sum()
    avg_pe = valid_pe['pe_ratio'].mean()
    median_pe = valid_pe['pe_ratio'].median()
    min_pe = valid_pe['pe_ratio'].min()
    max_pe = valid_pe['pe_ratio'].max()

    # Find extremes
    lowest_pe_company = valid_pe.loc[valid_pe['pe_ratio'].idxmin()]
    highest_pe_company = valid_pe.loc[valid_pe['pe_ratio'].idxmax()]
    largest_income = valid_income.loc[valid_income['ttm_net_income'].idxmax()]

    summary = f"""
[bold white]Summary Statistics[/bold white]

[cyan]Aggregate Metrics:[/cyan]
  Total Market Cap:     {format_currency(total_market_cap)}
  Total TTM Net Income: {format_currency(total_net_income)}
  Aggregate P/E:        {format_pe(total_market_cap / total_net_income if total_net_income else None)}

[cyan]P/E Ratio Statistics:[/cyan]
  Average P/E:          {format_pe(avg_pe)}
  Median P/E:           {format_pe(median_pe)}
  Min P/E:              {format_pe(min_pe)} ({lowest_pe_company['ticker']} - {lowest_pe_company['name'][:20]})
  Max P/E:              {format_pe(max_pe)} ({highest_pe_company['ticker']} - {highest_pe_company['name'][:20]})

[cyan]Notable Companies:[/cyan]
  Highest Net Income:   {largest_income['ticker']} - {format_currency(largest_income['ttm_net_income'])}
  Most Expensive (P/E): {highest_pe_company['ticker']} ({format_pe(highest_pe_company['pe_ratio'])})
  Most Affordable (P/E):{lowest_pe_company['ticker']} ({format_pe(lowest_pe_company['pe_ratio'])})
"""

    console.print(Panel(summary, title="Valuation Summary", border_style="green"))


def display_sector_analysis(df: pd.DataFrame) -> None:
    """Display sector-level analysis."""
    console = Console()

    # Group by sector
    sector_stats = df.groupby('sector').agg({
        'market_cap': 'sum',
        'ttm_net_income': 'sum',
        'ticker': 'count'
    }).rename(columns={'ticker': 'companies'})

    # Calculate sector P/E
    sector_stats['sector_pe'] = sector_stats['market_cap'] / sector_stats['ttm_net_income']
    sector_stats = sector_stats.sort_values('market_cap', ascending=False)

    table = Table(
        title="Sector Analysis",
        show_header=True,
        header_style="bold blue",
        border_style="blue",
    )

    table.add_column("Sector", style="white", width=22)
    table.add_column("Companies", justify="center", width=10)
    table.add_column("Total Mkt Cap", justify="right", style="green", width=14)
    table.add_column("Total Net Income", justify="right", style="yellow", width=14)
    table.add_column("Sector P/E", justify="right", style="magenta", width=10)

    for sector, row in sector_stats.iterrows():
        if sector == 'N/A':
            continue
        table.add_row(
            str(sector)[:22],
            str(int(row['companies'])),
            format_currency(row['market_cap']),
            format_currency(row['ttm_net_income']),
            format_pe(row['sector_pe']),
        )

    console.print(table)


def run_analysis():
    """Run the complete valuation analysis."""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Company Valuation Analysis[/bold cyan]\n"
        "Market Cap vs TTM Net Income (Last 4 Quarters)",
        border_style="cyan"
    ))

    # Fetch and analyze data
    df = analyze_top_companies()

    # Display results
    console.print()
    display_valuation_table(df)
    console.print()
    display_summary_stats(df)
    console.print()
    display_sector_analysis(df)

    # Return dataframe for further analysis
    return df


if __name__ == "__main__":
    run_analysis()
