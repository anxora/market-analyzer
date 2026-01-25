"""
Technical indicators for intraday trading: RSI, MACD, VWAP.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IndicatorValues:
    """Container for current indicator values."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    vwap: float
    price: float
    timestamp: pd.Timestamp


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series of RSI values
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use exponential moving average for smoother RSI
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD Line = 12-period EMA - 26-period EMA
    Signal Line = 9-period EMA of MACD Line
    Histogram = MACD Line - Signal Line

    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    reset_daily: bool = True
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        volume: Series of volume
        reset_daily: Reset VWAP calculation each trading day

    Returns:
        Series of VWAP values
    """
    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume

    if reset_daily and isinstance(high.index, pd.DatetimeIndex):
        # Group by date and calculate cumulative within each day
        dates = high.index.date
        cumulative_tp_vol = tp_volume.groupby(dates).cumsum()
        cumulative_vol = volume.groupby(dates).cumsum()
    else:
        cumulative_tp_vol = tp_volume.cumsum()
        cumulative_vol = volume.cumsum()

    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns:
        Tuple of (Upper Band, Middle Band (SMA), Lower Band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


class TechnicalAnalyzer:
    """
    Technical analysis calculator for intraday trading.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for a DataFrame with OHLCV data.

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with added indicator columns
        """
        result = df.copy()

        # RSI
        result['RSI'] = calculate_rsi(df['Close'], self.rsi_period)

        # MACD
        macd, signal, hist = calculate_macd(
            df['Close'],
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Histogram'] = hist

        # VWAP
        result['VWAP'] = calculate_vwap(
            df['High'], df['Low'], df['Close'], df['Volume']
        )

        # Additional useful indicators
        result['EMA_9'] = calculate_ema(df['Close'], 9)
        result['EMA_21'] = calculate_ema(df['Close'], 21)
        result['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])

        return result

    def get_current_values(self, df: pd.DataFrame) -> IndicatorValues:
        """
        Get the most recent indicator values.

        Args:
            df: DataFrame with OHLCV data (will calculate indicators)

        Returns:
            IndicatorValues with current readings
        """
        analyzed = self.analyze(df)
        latest = analyzed.iloc[-1]

        return IndicatorValues(
            rsi=latest['RSI'],
            macd=latest['MACD'],
            macd_signal=latest['MACD_Signal'],
            macd_histogram=latest['MACD_Histogram'],
            vwap=latest['VWAP'],
            price=latest['Close'],
            timestamp=analyzed.index[-1]
        )


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    print("Testing indicators with NVDA 5-minute data...")
    nvda = yf.download("NVDA", period="5d", interval="5m", progress=False)

    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze(nvda)

    print(f"\nLatest values:")
    print(f"  Price: ${result['Close'].iloc[-1]:.2f}")
    print(f"  RSI: {result['RSI'].iloc[-1]:.1f}")
    print(f"  MACD: {result['MACD'].iloc[-1]:.3f}")
    print(f"  MACD Signal: {result['MACD_Signal'].iloc[-1]:.3f}")
    print(f"  MACD Histogram: {result['MACD_Histogram'].iloc[-1]:.3f}")
    print(f"  VWAP: ${result['VWAP'].iloc[-1]:.2f}")
    print(f"  ATR: ${result['ATR'].iloc[-1]:.2f}")
