"""
Signal generation for intraday trading based on RSI, MACD, and VWAP.
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

from .indicators import TechnicalAnalyzer, IndicatorValues


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class TradeSignal:
    """Complete trade signal with all details."""
    signal: Signal
    strength: SignalStrength
    price: float
    timestamp: datetime
    reasons: List[str]
    indicators: IndicatorValues

    # Suggested order parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 1.0  # Percentage of available capital

    def __str__(self) -> str:
        return (
            f"{self.signal.value} ({self.strength.name}) @ ${self.price:.2f}\n"
            f"  Reasons: {', '.join(self.reasons)}\n"
            f"  RSI: {self.indicators.rsi:.1f} | MACD: {self.indicators.macd:.3f} | "
            f"VWAP: ${self.indicators.vwap:.2f}"
        )


class SignalGenerator:
    """
    Generate trading signals based on RSI, MACD, and VWAP indicators.

    Strategy:
    - RSI: Oversold (<30) = bullish, Overbought (>70) = bearish
    - MACD: Histogram positive & rising = bullish, negative & falling = bearish
    - VWAP: Price above VWAP = bullish, below = bearish

    BUY when 2+ indicators are bullish
    SELL when 2+ indicators are bearish
    """

    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vwap_threshold_pct: float = 0.1,  # Price must be X% from VWAP
        require_confluence: int = 2,  # Minimum bullish/bearish signals needed
        atr_stop_multiplier: float = 1.5,  # Stop loss = ATR * multiplier
        atr_target_multiplier: float = 2.0,  # Take profit = ATR * multiplier
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vwap_threshold_pct = vwap_threshold_pct
        self.require_confluence = require_confluence
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier

        self.analyzer = TechnicalAnalyzer(
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal
        )

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """
        Generate a trading signal from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            TradeSignal with recommendation
        """
        # Calculate all indicators
        analyzed = self.analyzer.analyze(df)

        # Get current and previous values
        current = analyzed.iloc[-1]
        previous = analyzed.iloc[-2] if len(analyzed) > 1 else current

        price = current['Close']
        rsi = current['RSI']
        macd = current['MACD']
        macd_signal = current['MACD_Signal']
        macd_hist = current['MACD_Histogram']
        macd_hist_prev = previous['MACD_Histogram']
        vwap = current['VWAP']
        atr = current['ATR']

        # Collect bullish/bearish signals
        bullish_signals = []
        bearish_signals = []

        # RSI Analysis
        if rsi < self.rsi_oversold:
            bullish_signals.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            bearish_signals.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:
            bullish_signals.append(f"RSI bullish zone ({rsi:.1f})")
        elif rsi > 55:
            bearish_signals.append(f"RSI bearish zone ({rsi:.1f})")

        # MACD Analysis
        macd_bullish_cross = macd > macd_signal and previous['MACD'] <= previous['MACD_Signal']
        macd_bearish_cross = macd < macd_signal and previous['MACD'] >= previous['MACD_Signal']

        if macd_bullish_cross:
            bullish_signals.append("MACD bullish crossover")
        elif macd_bearish_cross:
            bearish_signals.append("MACD bearish crossover")
        elif macd_hist > 0 and macd_hist > macd_hist_prev:
            bullish_signals.append(f"MACD histogram rising ({macd_hist:.3f})")
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            bearish_signals.append(f"MACD histogram falling ({macd_hist:.3f})")

        # VWAP Analysis
        vwap_distance_pct = ((price - vwap) / vwap) * 100

        if price > vwap * (1 + self.vwap_threshold_pct / 100):
            bullish_signals.append(f"Price above VWAP (+{vwap_distance_pct:.2f}%)")
        elif price < vwap * (1 - self.vwap_threshold_pct / 100):
            bearish_signals.append(f"Price below VWAP ({vwap_distance_pct:.2f}%)")

        # Determine signal based on confluence
        bullish_count = len(bullish_signals)
        bearish_count = len(bearish_signals)

        if bullish_count >= self.require_confluence and bullish_count > bearish_count:
            signal = Signal.BUY
            reasons = bullish_signals
            strength = self._calculate_strength(bullish_count)
            stop_loss = price - (atr * self.atr_stop_multiplier)
            take_profit = price + (atr * self.atr_target_multiplier)
        elif bearish_count >= self.require_confluence and bearish_count > bullish_count:
            signal = Signal.SELL
            reasons = bearish_signals
            strength = self._calculate_strength(bearish_count)
            stop_loss = price + (atr * self.atr_stop_multiplier)
            take_profit = price - (atr * self.atr_target_multiplier)
        else:
            signal = Signal.HOLD
            reasons = ["No clear confluence"] + bullish_signals + bearish_signals
            strength = SignalStrength.WEAK
            stop_loss = None
            take_profit = None

        # Calculate position size based on signal strength
        position_size = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.50,
            SignalStrength.STRONG: 1.0
        }[strength]

        indicator_values = IndicatorValues(
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            vwap=vwap,
            price=price,
            timestamp=analyzed.index[-1]
        )

        return TradeSignal(
            signal=signal,
            strength=strength,
            price=price,
            timestamp=datetime.now(),
            reasons=reasons,
            indicators=indicator_values,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size
        )

    def _calculate_strength(self, signal_count: int) -> SignalStrength:
        """Determine signal strength based on number of confirming indicators."""
        if signal_count >= 3:
            return SignalStrength.STRONG
        elif signal_count >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def analyze_trend(self, df: pd.DataFrame) -> dict:
        """
        Analyze overall trend using multiple timeframes.

        Returns:
            Dict with trend analysis
        """
        analyzed = self.analyzer.analyze(df)
        current = analyzed.iloc[-1]

        # EMA trend
        ema_9 = current['EMA_9']
        ema_21 = current['EMA_21']
        price = current['Close']

        trend = "BULLISH" if ema_9 > ema_21 and price > ema_9 else \
                "BEARISH" if ema_9 < ema_21 and price < ema_9 else \
                "NEUTRAL"

        return {
            'trend': trend,
            'price': price,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': current['RSI'],
            'macd_histogram': current['MACD_Histogram'],
            'above_vwap': price > current['VWAP'],
            'volatility_atr': current['ATR'],
            'volatility_pct': (current['ATR'] / price) * 100
        }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing signal generation with NVDA...")
    nvda = yf.download("NVDA", period="5d", interval="5m", progress=False)

    generator = SignalGenerator()
    signal = generator.generate_signal(nvda)

    print(f"\n=== Current Signal ===")
    print(signal)

    if signal.stop_loss:
        print(f"\n  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Position Size: {signal.position_size_pct * 100:.0f}%")

    print(f"\n=== Trend Analysis ===")
    trend = generator.analyze_trend(nvda)
    for key, value in trend.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
