"""
Multi-provider data fetching with automatic fallback.

Supports multiple data sources with automatic failover:
1. Yahoo Finance (yfinance) - Primary, no API key required
2. Alpha Vantage - Backup #1, free tier available
3. Financial Modeling Prep (FMP) - Backup #2, free tier available
"""

import os
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class RateLimitError(DataProviderError):
    """Raised when API rate limit is hit."""
    pass


class DataProvider(ABC):
    """Abstract base class for data providers."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    def get_quote(self, ticker: str) -> Dict:
        """Get current quote data for a ticker."""
        pass

    @abstractmethod
    def get_company_info(self, ticker: str) -> Dict:
        """Get company information."""
        pass

    @abstractmethod
    def get_income_statement(self, ticker: str, quarterly: bool = True) -> pd.DataFrame:
        """Get income statement data."""
        pass

    @abstractmethod
    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (has API key if needed)."""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""

    name = "yahoo_finance"
    requires_api_key = False

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def get_quote(self, ticker: str) -> Dict:
        try:
            stock = self.yf.Ticker(ticker)
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
                'provider': self.name,
            }
        except Exception as e:
            if 'RateLimit' in str(type(e).__name__) or 'Too Many Requests' in str(e):
                raise RateLimitError(f"Yahoo Finance rate limit: {e}")
            raise DataProviderError(f"Yahoo Finance error: {e}")

    def get_company_info(self, ticker: str) -> Dict:
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            return {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'employees': info.get('fullTimeEmployees'),
                'description': info.get('longBusinessSummary'),
                'website': info.get('website'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'provider': self.name,
            }
        except Exception as e:
            if 'RateLimit' in str(type(e).__name__) or 'Too Many Requests' in str(e):
                raise RateLimitError(f"Yahoo Finance rate limit: {e}")
            raise DataProviderError(f"Yahoo Finance error: {e}")

    def get_income_statement(self, ticker: str, quarterly: bool = True) -> pd.DataFrame:
        try:
            stock = self.yf.Ticker(ticker)
            if quarterly:
                return stock.quarterly_income_stmt
            return stock.income_stmt
        except Exception as e:
            if 'RateLimit' in str(type(e).__name__):
                raise RateLimitError(f"Yahoo Finance rate limit: {e}")
            raise DataProviderError(f"Yahoo Finance error: {e}")

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        try:
            stock = self.yf.Ticker(ticker)
            return stock.history(period=period)
        except Exception as e:
            if 'RateLimit' in str(type(e).__name__):
                raise RateLimitError(f"Yahoo Finance rate limit: {e}")
            raise DataProviderError(f"Yahoo Finance error: {e}")


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""

    name = "alpha_vantage"
    requires_api_key = True
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self._available = bool(self.api_key)

    def is_available(self) -> bool:
        return self._available

    def _make_request(self, params: Dict) -> Dict:
        params['apikey'] = self.api_key
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'Note' in data:  # Rate limit message
                raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")
            if 'Error Message' in data:
                raise DataProviderError(f"Alpha Vantage error: {data['Error Message']}")

            return data
        except requests.RequestException as e:
            raise DataProviderError(f"Alpha Vantage request error: {e}")

    def get_quote(self, ticker: str) -> Dict:
        data = self._make_request({
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker
        })
        quote = data.get('Global Quote', {})
        return {
            'ticker': ticker,
            'price': float(quote.get('05. price', 0)),
            'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
            'volume': int(quote.get('06. volume', 0)),
            'market_cap': None,  # Not available in this endpoint
            'pe_ratio': None,
            'dividend_yield': None,
            'fifty_two_week_high': float(quote.get('03. high', 0)),
            'fifty_two_week_low': float(quote.get('04. low', 0)),
            'provider': self.name,
        }

    def get_company_info(self, ticker: str) -> Dict:
        data = self._make_request({
            'function': 'OVERVIEW',
            'symbol': ticker
        })
        return {
            'ticker': ticker,
            'name': data.get('Name', ticker),
            'sector': data.get('Sector'),
            'industry': data.get('Industry'),
            'country': data.get('Country'),
            'employees': int(data.get('FullTimeEmployees', 0)) if data.get('FullTimeEmployees') else None,
            'description': data.get('Description'),
            'website': None,  # Not available
            'market_cap': int(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') else None,
            'enterprise_value': None,
            'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio') and data.get('PERatio') != 'None' else None,
            'dividend_yield': float(data.get('DividendYield', 0)) if data.get('DividendYield') else None,
            'provider': self.name,
        }

    def get_income_statement(self, ticker: str, quarterly: bool = True) -> pd.DataFrame:
        function = 'INCOME_STATEMENT'
        data = self._make_request({
            'function': function,
            'symbol': ticker
        })

        reports = data.get('quarterlyReports' if quarterly else 'annualReports', [])
        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)
        df = df.set_index('fiscalDateEnding')
        return df.T

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        # Map period to Alpha Vantage outputsize
        outputsize = 'full' if period in ['1y', '2y', '5y', 'max'] else 'compact'

        data = self._make_request({
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': outputsize
        })

        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return pd.DataFrame()

        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        df = df.sort_index()

        # Filter by period
        days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
        days = days_map.get(period, 365)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

        return df


class FinancialModelingPrepProvider(DataProvider):
    """Financial Modeling Prep (FMP) data provider - Updated for new stable API (2025+)."""

    name = "fmp"
    requires_api_key = True
    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        self._available = bool(self.api_key)

    def is_available(self) -> bool:
        return self._available

    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        params = params or {}
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)

            # Handle payment required (free tier limitation)
            if response.status_code == 402:
                raise DataProviderError(f"FMP endpoint '{endpoint}' requires paid subscription")

            response.raise_for_status()
            data = response.json()

            # Handle string error messages
            if isinstance(data, str) and ('Restricted' in data or 'Error' in data):
                raise DataProviderError(f"FMP error: {data}")

            if isinstance(data, dict) and 'Error Message' in data:
                raise DataProviderError(f"FMP error: {data['Error Message']}")

            return data
        except requests.RequestException as e:
            raise DataProviderError(f"FMP request error: {e}")

    def get_quote(self, ticker: str) -> Dict:
        data = self._make_request("quote", {'symbol': ticker})
        if not data or len(data) == 0:
            raise DataProviderError(f"No data for {ticker}")

        quote = data[0]
        return {
            'ticker': ticker,
            'price': quote.get('price'),
            'change_percent': quote.get('changePercentage'),
            'volume': quote.get('volume'),
            'market_cap': quote.get('marketCap'),
            'pe_ratio': quote.get('pe'),
            'dividend_yield': None,
            'fifty_two_week_high': quote.get('yearHigh'),
            'fifty_two_week_low': quote.get('yearLow'),
            'provider': self.name,
        }

    def get_company_info(self, ticker: str) -> Dict:
        data = self._make_request("profile", {'symbol': ticker})
        if not data or len(data) == 0:
            raise DataProviderError(f"No data for {ticker}")

        profile = data[0]
        return {
            'ticker': ticker,
            'name': profile.get('companyName', ticker),
            'sector': profile.get('sector'),
            'industry': profile.get('industry'),
            'country': profile.get('country'),
            'employees': profile.get('fullTimeEmployees'),
            'description': profile.get('description'),
            'website': profile.get('website'),
            'market_cap': profile.get('marketCap'),
            'enterprise_value': None,
            'pe_ratio': profile.get('pe'),
            'dividend_yield': profile.get('lastDividend'),
            'provider': self.name,
        }

    def get_income_statement(self, ticker: str, quarterly: bool = True) -> pd.DataFrame:
        period = 'quarter' if quarterly else 'annual'
        data = self._make_request("income-statement", {
            'symbol': ticker,
            'period': period,
            'limit': 8
        })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df = df.set_index('date')
        return df.T

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        data = self._make_request("historical-price-eod/full", {'symbol': ticker})

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_index()

        # Filter by period
        days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
        days = days_map.get(period, 365)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

        return df


class MultiProviderFetcher:
    """
    Multi-provider data fetcher with automatic fallback.

    Tries providers in order until one succeeds.
    """

    def __init__(self, providers: List[DataProvider] = None):
        if providers is None:
            # Default provider order
            self.providers = [
                YahooFinanceProvider(),
                AlphaVantageProvider(),
                FinancialModelingPrepProvider(),
            ]
        else:
            self.providers = providers

        # Filter to available providers
        self.available_providers = [p for p in self.providers if p.is_available()]

        if not self.available_providers:
            print("Warning: No data providers available. Install yfinance or set API keys.")

    def _try_providers(self, method_name: str, *args, **kwargs) -> Any:
        """Try each provider until one succeeds."""
        last_error = None

        for provider in self.available_providers:
            try:
                method = getattr(provider, method_name)
                result = method(*args, **kwargs)
                return result
            except RateLimitError as e:
                print(f"[{provider.name}] Rate limited, trying next provider...")
                last_error = e
                continue
            except DataProviderError as e:
                print(f"[{provider.name}] Error: {e}, trying next provider...")
                last_error = e
                continue
            except Exception as e:
                print(f"[{provider.name}] Unexpected error: {e}, trying next provider...")
                last_error = e
                continue

        # All providers failed
        raise DataProviderError(f"All providers failed. Last error: {last_error}")

    def get_quote(self, ticker: str) -> Dict:
        """Get quote with automatic fallback."""
        return self._try_providers('get_quote', ticker)

    def get_company_info(self, ticker: str) -> Dict:
        """Get company info with automatic fallback."""
        return self._try_providers('get_company_info', ticker)

    def get_income_statement(self, ticker: str, quarterly: bool = True) -> pd.DataFrame:
        """Get income statement with automatic fallback."""
        return self._try_providers('get_income_statement', ticker, quarterly)

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical prices with automatic fallback."""
        return self._try_providers('get_historical_prices', ticker, period)

    def list_providers(self) -> List[str]:
        """List available providers."""
        return [p.name for p in self.available_providers]


# Global fetcher instance
_fetcher: Optional[MultiProviderFetcher] = None


def get_fetcher() -> MultiProviderFetcher:
    """Get or create the global fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = MultiProviderFetcher()
    return _fetcher


# Convenience functions
def fetch_quote(ticker: str) -> Dict:
    """Fetch quote with automatic fallback."""
    return get_fetcher().get_quote(ticker)


def fetch_company_info(ticker: str) -> Dict:
    """Fetch company info with automatic fallback."""
    return get_fetcher().get_company_info(ticker)


def fetch_income_statement(ticker: str, quarterly: bool = True) -> pd.DataFrame:
    """Fetch income statement with automatic fallback."""
    return get_fetcher().get_income_statement(ticker, quarterly)


def fetch_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical prices with automatic fallback."""
    return get_fetcher().get_historical_prices(ticker, period)


if __name__ == "__main__":
    # Test the multi-provider fetcher
    fetcher = MultiProviderFetcher()
    print(f"Available providers: {fetcher.list_providers()}")

    try:
        quote = fetcher.get_quote('AAPL')
        print(f"\nAAPL Quote: ${quote['price']:.2f} (via {quote['provider']})")
    except DataProviderError as e:
        print(f"Error: {e}")
