"""Trading strategies for backtesting."""

from enum import Enum

import pandas as pd


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class VolatilityBreakoutStrategy:
    """Volatility Breakout Strategy.

    Buy when price breaks above: previous close + (k * previous range)
    Sell at end of day or when price breaks below entry - stop loss.

    This is a day trading strategy that captures intraday volatility moves.
    """

    def __init__(self, k: float = 0.5, stop_loss_pct: float = 0.02):
        """
        Args:
            k: Multiplier for the breakout range (0.5 = 50% of previous day range)
            stop_loss_pct: Stop loss percentage (0.02 = 2%)
        """
        self.k = k
        self.stop_loss_pct = stop_loss_pct

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on volatility breakout.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added signal columns
        """
        df = df.copy()

        # Calculate previous day's range
        df["Prev_Range"] = (df["High"].shift(1) - df["Low"].shift(1))
        df["Prev_Close"] = df["Close"].shift(1)

        # Breakout level: previous close + k * previous range
        df["Breakout_Level"] = df["Prev_Close"] + (self.k * df["Prev_Range"])

        df["Signal"] = Signal.HOLD.value

        # Buy when price breaks above the breakout level
        df.loc[df["High"] > df["Breakout_Level"], "Signal"] = Signal.BUY.value

        # For daily data, we sell at close (end of day)
        # In intraday, we would sell at a target or stop loss

        return df
