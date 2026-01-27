"""Trading strategies for backtesting."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    signal: Signal
    price: float
    timestamp: pd.Timestamp
    reason: str


class SMAStrategy:
    """Simple Moving Average Crossover Strategy.

    Generates buy signals when short SMA crosses above long SMA,
    and sell signals when short SMA crosses below long SMA.
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on SMA crossover.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added signal columns
        """
        df = df.copy()

        df["SMA_short"] = df["Close"].rolling(window=self.short_window).mean()
        df["SMA_long"] = df["Close"].rolling(window=self.long_window).mean()

        df["Signal"] = Signal.HOLD.value

        # Buy when short SMA crosses above long SMA
        df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = Signal.BUY.value

        # Sell when short SMA crosses below long SMA
        df.loc[df["SMA_short"] < df["SMA_long"], "Signal"] = Signal.SELL.value

        return df


class RSIStrategy:
    """RSI (Relative Strength Index) Strategy.

    Buy when RSI is oversold (< 30), sell when overbought (> 70).
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on RSI.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added signal columns
        """
        df = df.copy()

        df["RSI"] = self.calculate_rsi(df["Close"])

        df["Signal"] = Signal.HOLD.value

        # Buy when RSI is oversold
        df.loc[df["RSI"] < self.oversold, "Signal"] = Signal.BUY.value

        # Sell when RSI is overbought
        df.loc[df["RSI"] > self.overbought, "Signal"] = Signal.SELL.value

        return df


class MomentumStrategy:
    """Simple Momentum Strategy.

    Buy when price increases by threshold %, sell when decreases.
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on momentum.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added signal columns
        """
        df = df.copy()

        df["Momentum"] = df["Close"].pct_change(periods=self.lookback)

        df["Signal"] = Signal.HOLD.value

        # Buy on positive momentum
        df.loc[df["Momentum"] > self.threshold, "Signal"] = Signal.BUY.value

        # Sell on negative momentum
        df.loc[df["Momentum"] < -self.threshold, "Signal"] = Signal.SELL.value

        return df
