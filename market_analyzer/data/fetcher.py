"""Data fetcher for historical stock prices."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_historical_data(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    period: str = "1mo"
) -> pd.DataFrame:
    """Fetch historical stock data using yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date for data
        end_date: End date for data
        period: Period string if dates not specified ('1d', '5d', '1mo', etc.)

    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        df = ticker.history(start=start_date, end=end_date)
    else:
        df = ticker.history(period=period)

    return df


def fetch_yesterday_data(symbol: str) -> pd.DataFrame:
    """Fetch data for yesterday's trading session.

    Args:
        symbol: Stock ticker symbol

    Returns:
        DataFrame with yesterday's OHLCV data
    """
    today = datetime.now()
    start = today - timedelta(days=7)  # Get a week to ensure we have trading days

    df = fetch_historical_data(symbol, start_date=start, end_date=today)

    if len(df) >= 1:
        return df

    return df


def fetch_intraday_data(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday data for backtesting.

    Args:
        symbol: Stock ticker symbol
        period: Period ('1d', '5d', etc.)
        interval: Data interval ('1m', '5m', '15m', '30m', '1h')

    Returns:
        DataFrame with intraday OHLCV data
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df
