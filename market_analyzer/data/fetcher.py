"""Data fetching utilities using yfinance."""

import yfinance as yf
import pandas as pd
from typing import Optional


def get_company_info(ticker: str) -> dict:
    """Get company information including market cap and financials."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'ticker': ticker,
        'name': info.get('longName', info.get('shortName', ticker)),
        'market_cap': info.get('marketCap', 0),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'country': info.get('country', 'N/A'),
        'currency': info.get('currency', 'USD'),
    }


def get_quarterly_financials(ticker: str) -> pd.DataFrame:
    """Get quarterly income statement data."""
    stock = yf.Ticker(ticker)
    return stock.quarterly_financials


def get_trailing_net_income(ticker: str, quarters: int = 4) -> Optional[float]:
    """
    Get trailing net income for the last N quarters.

    Args:
        ticker: Stock ticker symbol
        quarters: Number of quarters to sum (default 4 for TTM)

    Returns:
        Sum of net income for the specified quarters, or None if data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.quarterly_income_stmt

        if financials.empty:
            return None

        # Look for Net Income row
        net_income_rows = ['Net Income', 'Net Income Common Stockholders',
                          'Net Income From Continuing Operations']

        net_income = None
        for row_name in net_income_rows:
            if row_name in financials.index:
                net_income = financials.loc[row_name]
                break

        if net_income is None:
            return None

        # Get the last N quarters and sum them
        available_quarters = min(quarters, len(net_income))
        ttm_net_income = net_income.iloc[:available_quarters].sum()

        return float(ttm_net_income)
    except Exception as e:
        print(f"Error fetching net income for {ticker}: {e}")
        return None


def get_valuation_data(ticker: str) -> dict:
    """
    Get complete valuation data for a company.

    Returns dict with market cap, TTM net income, and P/E ratio.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get basic info
        market_cap = info.get('marketCap', 0)
        name = info.get('longName', info.get('shortName', ticker))
        sector = info.get('sector', 'N/A')

        # Get TTM net income
        ttm_net_income = get_trailing_net_income(ticker, quarters=4)

        # Calculate P/E ratio
        pe_ratio = None
        if ttm_net_income and ttm_net_income > 0:
            pe_ratio = market_cap / ttm_net_income

        return {
            'ticker': ticker,
            'name': name,
            'sector': sector,
            'market_cap': market_cap,
            'ttm_net_income': ttm_net_income,
            'pe_ratio': pe_ratio,
        }
    except Exception as e:
        print(f"Error fetching valuation data for {ticker}: {e}")
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'N/A',
            'market_cap': None,
            'ttm_net_income': None,
            'pe_ratio': None,
        }
