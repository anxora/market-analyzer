"""
Data fetching utilities with automatic fallback.

Uses yfinance as primary provider with automatic fallback to
Alpha Vantage and Financial Modeling Prep when rate limited.
"""

import pandas as pd
from typing import Optional

# Try to import providers for fallback support
try:
    from .providers import (
        MultiProviderFetcher,
        get_fetcher,
        DataProviderError,
        RateLimitError,
    )
    HAS_FALLBACK = True
except ImportError:
    HAS_FALLBACK = False

# Primary provider (yfinance)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def _get_yf_ticker(ticker: str):
    """Get yfinance Ticker object."""
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    return yf.Ticker(ticker)


def get_company_info(ticker: str, use_fallback: bool = True) -> dict:
    """
    Get company information including market cap and financials.

    Args:
        ticker: Stock ticker symbol
        use_fallback: If True, try alternative providers on failure

    Returns:
        Dict with company information
    """
    # Try yfinance first
    try:
        stock = _get_yf_ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', ticker)),
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'provider': 'yahoo_finance',
        }
    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            raise

        # Try fallback providers
        try:
            fetcher = get_fetcher()
            data = fetcher.get_company_info(ticker)
            return {
                'ticker': ticker,
                'name': data.get('name', ticker),
                'market_cap': data.get('market_cap', 0),
                'sector': data.get('sector', 'N/A'),
                'industry': data.get('industry', 'N/A'),
                'country': data.get('country', 'N/A'),
                'currency': 'USD',
                'provider': data.get('provider', 'fallback'),
            }
        except DataProviderError:
            raise


def get_quarterly_financials(ticker: str, use_fallback: bool = True) -> pd.DataFrame:
    """
    Get quarterly income statement data.

    Args:
        ticker: Stock ticker symbol
        use_fallback: If True, try alternative providers on failure

    Returns:
        DataFrame with quarterly financials
    """
    try:
        stock = _get_yf_ticker(ticker)
        return stock.quarterly_financials
    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            raise

        try:
            fetcher = get_fetcher()
            return fetcher.get_income_statement(ticker, quarterly=True)
        except DataProviderError:
            return pd.DataFrame()


def get_trailing_net_income(ticker: str, quarters: int = 4, use_fallback: bool = True) -> Optional[float]:
    """
    Get trailing net income for the last N quarters.

    Args:
        ticker: Stock ticker symbol
        quarters: Number of quarters to sum (default 4 for TTM)
        use_fallback: If True, try alternative providers on failure

    Returns:
        Sum of net income for the specified quarters, or None if data unavailable
    """
    try:
        stock = _get_yf_ticker(ticker)
        financials = stock.quarterly_income_stmt

        if financials.empty:
            raise ValueError("No financial data available")

        # Look for Net Income row
        net_income_rows = ['Net Income', 'Net Income Common Stockholders',
                          'Net Income From Continuing Operations']

        net_income = None
        for row_name in net_income_rows:
            if row_name in financials.index:
                net_income = financials.loc[row_name]
                break

        if net_income is None:
            raise ValueError("Net income not found in financials")

        # Get the last N quarters and sum them
        available_quarters = min(quarters, len(net_income))
        ttm_net_income = net_income.iloc[:available_quarters].sum()

        return float(ttm_net_income)

    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            return None

        # Try fallback
        try:
            fetcher = get_fetcher()
            df = fetcher.get_income_statement(ticker, quarterly=True)

            if df.empty:
                return None

            # Try to find net income in the fallback data
            net_income_keys = ['netIncome', 'Net Income', 'netIncomeCommon']
            for key in net_income_keys:
                if key in df.index:
                    values = df.loc[key].head(quarters)
                    values = pd.to_numeric(values, errors='coerce')
                    return float(values.sum())

            return None
        except DataProviderError:
            return None


def get_valuation_data(ticker: str, use_fallback: bool = True) -> dict:
    """
    Get complete valuation data for a company.

    Args:
        ticker: Stock ticker symbol
        use_fallback: If True, try alternative providers on failure

    Returns:
        Dict with market cap, TTM net income, and P/E ratio
    """
    try:
        stock = _get_yf_ticker(ticker)
        info = stock.info

        # Get basic info
        market_cap = info.get('marketCap', 0)
        name = info.get('longName', info.get('shortName', ticker))
        sector = info.get('sector', 'N/A')

        # Get TTM net income
        ttm_net_income = get_trailing_net_income(ticker, quarters=4, use_fallback=use_fallback)

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
            'provider': 'yahoo_finance',
        }

    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            return _empty_valuation_data(ticker)

        # Try fallback
        try:
            fetcher = get_fetcher()
            info = fetcher.get_company_info(ticker)

            market_cap = info.get('market_cap', 0)
            ttm_net_income = get_trailing_net_income(ticker, quarters=4, use_fallback=True)

            pe_ratio = None
            if ttm_net_income and ttm_net_income > 0 and market_cap:
                pe_ratio = market_cap / ttm_net_income

            return {
                'ticker': ticker,
                'name': info.get('name', ticker),
                'sector': info.get('sector', 'N/A'),
                'market_cap': market_cap,
                'ttm_net_income': ttm_net_income,
                'pe_ratio': pe_ratio,
                'provider': info.get('provider', 'fallback'),
            }
        except DataProviderError:
            return _empty_valuation_data(ticker)


def _empty_valuation_data(ticker: str) -> dict:
    """Return empty valuation data structure."""
    return {
        'ticker': ticker,
        'name': ticker,
        'sector': 'N/A',
        'market_cap': None,
        'ttm_net_income': None,
        'pe_ratio': None,
        'provider': None,
    }


def get_historical_prices(ticker: str, period: str = "1y", use_fallback: bool = True) -> pd.DataFrame:
    """
    Get historical price data.

    Args:
        ticker: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
        use_fallback: If True, try alternative providers on failure

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = _get_yf_ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            raise

        try:
            fetcher = get_fetcher()
            return fetcher.get_historical_prices(ticker, period)
        except DataProviderError:
            return pd.DataFrame()


def get_quote(ticker: str, use_fallback: bool = True) -> dict:
    """
    Get current quote data.

    Args:
        ticker: Stock ticker symbol
        use_fallback: If True, try alternative providers on failure

    Returns:
        Dict with current quote data
    """
    try:
        stock = _get_yf_ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'price': info.get('currentPrice', info.get('regularMarketPrice')),
            'change_percent': info.get('regularMarketChangePercent'),
            'volume': info.get('volume'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'provider': 'yahoo_finance',
        }
    except Exception as e:
        if not use_fallback or not HAS_FALLBACK:
            raise

        try:
            fetcher = get_fetcher()
            return fetcher.get_quote(ticker)
        except DataProviderError:
            raise
