"""
CAN SLIM Analysis Module - William O'Neil Method

CAN SLIM is an investment strategy developed by William O'Neil:
- C: Current quarterly earnings per share (EPS growth >= 25% YoY)
- A: Annual earnings growth (5-year growth rate >= 25%)
- N: New products, management, or price highs (52-week high proximity)
- S: Supply and demand (volume surge, low float preferred)
- L: Leader or laggard (Relative Strength >= 80)
- I: Institutional sponsorship (institutional ownership 20-60%)
- M: Market direction (overall market trend)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tickers import get_sp500_tickers, get_nasdaq100_tickers, get_all_us_tickers, get_growth_stocks

warnings.filterwarnings('ignore')


def calculate_eps_growth(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate quarterly and annual EPS growth.

    Returns:
        Tuple of (quarterly_growth_pct, annual_growth_pct)
    """
    try:
        stock = yf.Ticker(ticker)

        # Get quarterly earnings
        quarterly = stock.quarterly_income_stmt
        if quarterly.empty:
            return None, None

        # Get EPS data
        eps_rows = ['Basic EPS', 'Diluted EPS']
        eps_data = None
        for row in eps_rows:
            if row in quarterly.index:
                eps_data = quarterly.loc[row]
                break

        if eps_data is None or len(eps_data) < 5:
            return None, None

        # Quarterly growth (current vs same quarter last year)
        current_eps = eps_data.iloc[0]
        year_ago_eps = eps_data.iloc[4] if len(eps_data) > 4 else eps_data.iloc[-1]

        quarterly_growth = None
        if year_ago_eps and year_ago_eps != 0:
            quarterly_growth = ((current_eps - year_ago_eps) / abs(year_ago_eps)) * 100

        # Annual growth (TTM vs previous TTM)
        ttm_eps = eps_data.iloc[:4].sum() if len(eps_data) >= 4 else None
        prev_ttm_eps = eps_data.iloc[4:8].sum() if len(eps_data) >= 8 else None

        annual_growth = None
        if ttm_eps and prev_ttm_eps and prev_ttm_eps != 0:
            annual_growth = ((ttm_eps - prev_ttm_eps) / abs(prev_ttm_eps)) * 100

        return quarterly_growth, annual_growth
    except Exception:
        return None, None


def calculate_relative_strength(ticker: str, days: int = 252) -> Optional[float]:
    """
    Calculate Relative Strength (RS) rating compared to market.
    RS Rating 1-99, where 99 means outperforming 99% of stocks.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)

        # Get stock and SPY data
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

        if stock.empty or spy.empty or len(stock) < days:
            return None

        # Calculate returns
        stock_return = (stock['Close'].iloc[-1] / stock['Close'].iloc[-days]) - 1
        spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[-days]) - 1

        # Handle multi-index columns
        if isinstance(stock_return, pd.Series):
            stock_return = stock_return.iloc[0]
        if isinstance(spy_return, pd.Series):
            spy_return = spy_return.iloc[0]

        # Relative performance
        relative_perf = stock_return - spy_return

        # Convert to 1-99 scale (simplified)
        # Assuming normal distribution, map to percentile
        rs_rating = min(99, max(1, int(50 + relative_perf * 200)))

        return rs_rating
    except Exception:
        return None


def check_new_high(ticker: str, threshold_pct: float = 5.0) -> Tuple[bool, Optional[float]]:
    """
    Check if stock is within threshold% of 52-week high.

    Returns:
        Tuple of (is_near_high, pct_from_high)
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty:
            return False, None

        current_price = hist['Close'].iloc[-1]
        high_52w = hist['High'].max()

        pct_from_high = ((high_52w - current_price) / high_52w) * 100
        is_near_high = pct_from_high <= threshold_pct

        return is_near_high, pct_from_high
    except Exception:
        return False, None


def check_volume_surge(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Check for volume surge (current vs 50-day average).

    Returns:
        Tuple of (volume_ratio, avg_volume_millions)
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty or len(hist) < 50:
            return None, None

        current_volume = hist['Volume'].iloc[-5:].mean()  # 5-day avg
        avg_volume_50 = hist['Volume'].iloc[-50:].mean()

        volume_ratio = current_volume / avg_volume_50 if avg_volume_50 > 0 else None
        avg_volume_m = avg_volume_50 / 1e6

        return volume_ratio, avg_volume_m
    except Exception:
        return None, None


def get_institutional_ownership(ticker: str) -> Optional[float]:
    """Get institutional ownership percentage."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        inst_ownership = info.get('heldPercentInstitutions', None)
        if inst_ownership:
            return inst_ownership * 100
        return None
    except Exception:
        return None


def analyze_stock_canslim(ticker: str) -> Dict:
    """
    Perform full CAN SLIM analysis on a single stock.
    """
    result = {
        'ticker': ticker,
        'name': None,
        'sector': None,
        'price': None,
        # C - Current quarterly earnings
        'quarterly_eps_growth': None,
        'c_score': 0,
        # A - Annual earnings
        'annual_eps_growth': None,
        'a_score': 0,
        # N - New high
        'near_52w_high': False,
        'pct_from_high': None,
        'n_score': 0,
        # S - Supply/demand
        'volume_ratio': None,
        'avg_volume_m': None,
        's_score': 0,
        # L - Leader
        'rs_rating': None,
        'l_score': 0,
        # I - Institutional
        'inst_ownership': None,
        'i_score': 0,
        # Total
        'canslim_score': 0,
        'passing_criteria': 0,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result['name'] = info.get('shortName', ticker)
        result['sector'] = info.get('sector', 'N/A')
        result['price'] = info.get('currentPrice', info.get('regularMarketPrice'))

        # C - Current quarterly earnings (>= 25% growth)
        q_growth, a_growth = calculate_eps_growth(ticker)
        result['quarterly_eps_growth'] = q_growth
        result['annual_eps_growth'] = a_growth
        if q_growth and q_growth >= 25:
            result['c_score'] = 1
        elif q_growth and q_growth >= 15:
            result['c_score'] = 0.5

        # A - Annual earnings (>= 25% growth)
        if a_growth and a_growth >= 25:
            result['a_score'] = 1
        elif a_growth and a_growth >= 15:
            result['a_score'] = 0.5

        # N - New high (within 5% of 52-week high)
        near_high, pct_from_high = check_new_high(ticker)
        result['near_52w_high'] = near_high
        result['pct_from_high'] = pct_from_high
        if near_high:
            result['n_score'] = 1
        elif pct_from_high and pct_from_high <= 15:
            result['n_score'] = 0.5

        # S - Supply/demand (volume above average)
        vol_ratio, avg_vol = check_volume_surge(ticker)
        result['volume_ratio'] = vol_ratio
        result['avg_volume_m'] = avg_vol
        if vol_ratio and vol_ratio >= 1.5:
            result['s_score'] = 1
        elif vol_ratio and vol_ratio >= 1.0:
            result['s_score'] = 0.5

        # L - Leader (RS rating >= 80)
        rs = calculate_relative_strength(ticker)
        result['rs_rating'] = rs
        if rs and rs >= 80:
            result['l_score'] = 1
        elif rs and rs >= 70:
            result['l_score'] = 0.5

        # I - Institutional (20-60% is ideal)
        inst = get_institutional_ownership(ticker)
        result['inst_ownership'] = inst
        if inst and 20 <= inst <= 70:
            result['i_score'] = 1
        elif inst and (10 <= inst < 20 or 70 < inst <= 85):
            result['i_score'] = 0.5

        # Calculate total score
        scores = [result['c_score'], result['a_score'], result['n_score'],
                  result['s_score'], result['l_score'], result['i_score']]
        result['canslim_score'] = sum(scores)
        result['passing_criteria'] = sum(1 for s in scores if s >= 1)

    except Exception as e:
        result['error'] = str(e)

    return result


def screen_stocks(tickers: List[str], min_score: float = 3.0) -> pd.DataFrame:
    """
    Screen a list of stocks using CAN SLIM criteria.
    """
    console = Console()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(tickers)} stocks...", total=len(tickers))

        # Use threading for faster analysis
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(analyze_stock_canslim, t): t for t in tickers}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.advance(task)

    df = pd.DataFrame(results)

    # Sort by CAN SLIM score
    df = df.sort_values('canslim_score', ascending=False)

    return df


def display_canslim_results(df: pd.DataFrame, top_n: int = 20) -> None:
    """Display CAN SLIM screening results."""
    console = Console()

    # Filter to top opportunities
    top_stocks = df.head(top_n)

    table = Table(
        title=f"CAN SLIM Opportunities - Top {top_n} Stocks",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Ticker", style="cyan bold", width=7)
    table.add_column("Company", width=20)
    table.add_column("Price", justify="right", width=9)
    table.add_column("C\nEPS Q%", justify="center", width=8)
    table.add_column("A\nEPS Y%", justify="center", width=8)
    table.add_column("N\n%High", justify="center", width=7)
    table.add_column("S\nVol", justify="center", width=6)
    table.add_column("L\nRS", justify="center", width=5)
    table.add_column("I\nInst%", justify="center", width=6)
    table.add_column("Score", justify="center", style="bold", width=6)

    for idx, row in top_stocks.iterrows():
        # Format values
        price = f"${row['price']:.2f}" if row['price'] else "N/A"

        # C - Quarterly EPS growth
        q_eps = row['quarterly_eps_growth']
        c_val = f"{q_eps:.0f}%" if q_eps else "N/A"
        c_style = "green" if row['c_score'] >= 1 else ("yellow" if row['c_score'] > 0 else "red")

        # A - Annual EPS growth
        a_eps = row['annual_eps_growth']
        a_val = f"{a_eps:.0f}%" if a_eps else "N/A"
        a_style = "green" if row['a_score'] >= 1 else ("yellow" if row['a_score'] > 0 else "red")

        # N - Near 52w high
        pct_high = row['pct_from_high']
        n_val = f"-{pct_high:.1f}%" if pct_high else "N/A"
        n_style = "green" if row['n_score'] >= 1 else ("yellow" if row['n_score'] > 0 else "red")

        # S - Volume
        vol = row['volume_ratio']
        s_val = f"{vol:.1f}x" if vol else "N/A"
        s_style = "green" if row['s_score'] >= 1 else ("yellow" if row['s_score'] > 0 else "red")

        # L - RS Rating
        rs = row['rs_rating']
        l_val = f"{rs}" if rs else "N/A"
        l_style = "green" if row['l_score'] >= 1 else ("yellow" if row['l_score'] > 0 else "red")

        # I - Institutional
        inst = row['inst_ownership']
        i_val = f"{inst:.0f}%" if inst else "N/A"
        i_style = "green" if row['i_score'] >= 1 else ("yellow" if row['i_score'] > 0 else "red")

        # Score
        score = row['canslim_score']
        score_style = "green bold" if score >= 4.5 else ("yellow bold" if score >= 3.5 else "white")

        table.add_row(
            str(len(top_stocks) - list(top_stocks.index).index(idx)),
            row['ticker'],
            (row['name'] or 'N/A')[:20],
            price,
            f"[{c_style}]{c_val}[/{c_style}]",
            f"[{a_style}]{a_val}[/{a_style}]",
            f"[{n_style}]{n_val}[/{n_style}]",
            f"[{s_style}]{s_val}[/{s_style}]",
            f"[{l_style}]{l_val}[/{l_style}]",
            f"[{i_style}]{i_val}[/{i_style}]",
            f"[{score_style}]{score:.1f}[/{score_style}]",
        )

    console.print(table)


def display_top_opportunities(df: pd.DataFrame) -> None:
    """Display detailed analysis of top opportunities."""
    console = Console()

    # Get top 5 with score >= 4
    top = df[df['canslim_score'] >= 3.5].head(10)

    if top.empty:
        console.print("[yellow]No stocks met the minimum CAN SLIM criteria.[/yellow]")
        return

    console.print("\n[bold cyan]Top CAN SLIM Opportunities - Detailed Analysis[/bold cyan]\n")

    for _, row in top.iterrows():
        # Build criteria summary
        criteria = []
        if row['c_score'] >= 1:
            criteria.append(f"[green]C: Strong Q EPS +{row['quarterly_eps_growth']:.0f}%[/green]")
        if row['a_score'] >= 1:
            criteria.append(f"[green]A: Strong Y EPS +{row['annual_eps_growth']:.0f}%[/green]")
        if row['n_score'] >= 1:
            criteria.append(f"[green]N: Near 52w High ({row['pct_from_high']:.1f}% off)[/green]")
        if row['s_score'] >= 1:
            criteria.append(f"[green]S: Volume Surge {row['volume_ratio']:.1f}x[/green]")
        if row['l_score'] >= 1:
            criteria.append(f"[green]L: Leader RS={row['rs_rating']}[/green]")
        if row['i_score'] >= 1:
            criteria.append(f"[green]I: Inst. {row['inst_ownership']:.0f}%[/green]")

        criteria_str = " | ".join(criteria) if criteria else "[yellow]Partial criteria met[/yellow]"

        panel_content = f"""
[bold white]{row['name']}[/bold white] ({row['sector']})
Price: [cyan]${row['price']:.2f}[/cyan]  |  Score: [bold yellow]{row['canslim_score']:.1f}/6[/bold yellow]

{criteria_str}
"""
        console.print(Panel(panel_content, title=f"[bold cyan]{row['ticker']}[/bold cyan]", border_style="cyan"))


def get_market_direction() -> Dict:
    """
    Analyze overall market direction (the 'M' in CAN SLIM).
    """
    console = Console()

    try:
        # Get market data
        spy = yf.Ticker('SPY')
        qqq = yf.Ticker('QQQ')

        spy_hist = spy.history(period="6mo")
        qqq_hist = qqq.history(period="6mo")

        # Calculate metrics
        spy_current = spy_hist['Close'].iloc[-1]
        spy_50ma = spy_hist['Close'].iloc[-50:].mean()
        spy_200ma = spy_hist['Close'].mean()

        qqq_current = qqq_hist['Close'].iloc[-1]
        qqq_50ma = qqq_hist['Close'].iloc[-50:].mean()
        qqq_200ma = qqq_hist['Close'].mean()

        # Trend analysis
        spy_above_50 = spy_current > spy_50ma
        spy_above_200 = spy_current > spy_200ma
        qqq_above_50 = qqq_current > qqq_50ma
        qqq_above_200 = qqq_current > qqq_200ma

        # Market health score
        health_score = sum([spy_above_50, spy_above_200, qqq_above_50, qqq_above_200])

        if health_score >= 4:
            market_status = "[bold green]CONFIRMED UPTREND[/bold green]"
            recommendation = "Ideal environment for CAN SLIM buying"
        elif health_score >= 2:
            market_status = "[bold yellow]UPTREND UNDER PRESSURE[/bold yellow]"
            recommendation = "Be selective, reduce position sizes"
        else:
            market_status = "[bold red]MARKET IN CORRECTION[/bold red]"
            recommendation = "Avoid new buys, protect capital"

        result = {
            'status': market_status,
            'health_score': health_score,
            'recommendation': recommendation,
            'spy_vs_50ma': ((spy_current / spy_50ma) - 1) * 100,
            'spy_vs_200ma': ((spy_current / spy_200ma) - 1) * 100,
            'qqq_vs_50ma': ((qqq_current / qqq_50ma) - 1) * 100,
            'qqq_vs_200ma': ((qqq_current / qqq_200ma) - 1) * 100,
        }

        return result
    except Exception as e:
        return {'error': str(e)}


def display_market_analysis(market: Dict) -> None:
    """Display market direction analysis."""
    console = Console()

    if 'error' in market:
        console.print(f"[red]Error analyzing market: {market['error']}[/red]")
        return

    content = f"""
[bold white]Market Direction (M in CAN SLIM)[/bold white]

Status: {market['status']}
Health Score: {market['health_score']}/4

[cyan]S&P 500 (SPY):[/cyan]
  vs 50-day MA: {market['spy_vs_50ma']:+.1f}%
  vs 200-day MA: {market['spy_vs_200ma']:+.1f}%

[cyan]NASDAQ 100 (QQQ):[/cyan]
  vs 50-day MA: {market['qqq_vs_50ma']:+.1f}%
  vs 200-day MA: {market['qqq_vs_200ma']:+.1f}%

[bold]Recommendation:[/bold] {market['recommendation']}
"""
    console.print(Panel(content, title="Market Analysis", border_style="magenta"))


def run_canslim_screening(full_scan: bool = True):
    """
    Run complete CAN SLIM screening.

    Args:
        full_scan: If True, scan all S&P 500 + NASDAQ stocks (~500+).
                   If False, scan only growth-focused subset (~150).
    """
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]William O'Neil CAN SLIM Screening[/bold cyan]\n"
        "Searching for high-growth opportunities in US Markets",
        border_style="cyan"
    ))

    # 1. Market Direction
    console.print("\n[bold]Step 1: Analyzing Market Direction...[/bold]")
    market = get_market_direction()
    display_market_analysis(market)

    # 2. Get tickers
    if full_scan:
        console.print("\n[bold]Step 2: Loading complete S&P 500 & NASDAQ listings...[/bold]")
        all_tickers = get_all_us_tickers()
    else:
        console.print("\n[bold]Step 2: Loading growth-focused stock list...[/bold]")
        all_tickers = get_growth_stocks()

    console.print(f"[cyan]Found {len(all_tickers)} stocks to analyze[/cyan]\n")

    # 3. Screen stocks
    df = screen_stocks(all_tickers)

    # 4. Display results
    console.print()
    display_canslim_results(df, top_n=30)

    # 5. Show top opportunities
    display_top_opportunities(df)

    # 6. Summary
    strong = len(df[df['canslim_score'] >= 4.5])
    moderate = len(df[(df['canslim_score'] >= 3.0) & (df['canslim_score'] < 4.5)])

    summary = f"""
[bold white]Screening Summary[/bold white]

Total Stocks Analyzed: {len(df)}
Strong Candidates (Score >= 4.5): [green]{strong}[/green]
Moderate Candidates (Score 3.0-4.5): [yellow]{moderate}[/yellow]

[dim]Criteria: C=Current EPS, A=Annual EPS, N=New High, S=Supply/Demand, L=Leader, I=Institutional[/dim]
[dim]Green = Passing | Yellow = Partial | Red = Failing[/dim]
"""
    console.print(Panel(summary, title="Summary", border_style="green"))

    return df


if __name__ == "__main__":
    run_canslim_screening()
