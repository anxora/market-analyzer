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


@dataclass
class BreakoutSignal:
    """Breakout trading signal."""
    signal: Signal
    direction: int  # 1 = LONG, -1 = SHORT, 0 = HOLD
    price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    momentum: float
    channel_high: float
    channel_low: float
    timestamp: datetime
    reason: str

    def __str__(self) -> str:
        dir_str = "LONG" if self.direction == 1 else "SHORT" if self.direction == -1 else "HOLD"
        return (
            f"{dir_str} @ ${self.price:.2f}\n"
            f"  Entry: ${self.entry_price:.2f} | Stop: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f}\n"
            f"  Momentum: {self.momentum*100:+.2f}% | ATR: ${self.atr:.2f}\n"
            f"  Channel: ${self.channel_low:.2f} - ${self.channel_high:.2f}\n"
            f"  Reason: {self.reason}"
        )


class BreakoutSignalGenerator:
    """
    Generate trading signals based on Donchian Channel Breakout + Momentum.

    Strategy (backtested +24% in 60 days):
    - LONG: Price breaks above 20-bar high + Momentum > 1%
    - SHORT: Price breaks below 20-bar low + Momentum < -1%
    - Stop Loss: 1.5 × ATR
    - Take Profit: 2.0 × ATR

    This strategy follows trends and works in both bullish and bearish markets.
    """

    def __init__(
        self,
        channel_period: int = 20,
        momentum_period: int = 10,
        momentum_threshold: float = 0.01,  # 1%
        atr_period: int = 14,
        stop_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.003,  # 0.3% risk per trade
    ):
        self.channel_period = channel_period
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.atr_period = atr_period
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.risk_per_trade = risk_per_trade

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators."""
        result = df.copy()

        # Donchian Channel (20-period high/low)
        result['Channel_High'] = df['High'].rolling(self.channel_period).max()
        result['Channel_Low'] = df['Low'].rolling(self.channel_period).min()

        # Momentum (price change over N periods)
        result['Momentum'] = df['Close'] / df['Close'].shift(self.momentum_period) - 1

        # ATR for stop loss calculation
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['ATR'] = true_range.ewm(span=self.atr_period, adjust=False).mean()

        return result

    def generate_signal(self, df: pd.DataFrame) -> BreakoutSignal:
        """
        Generate a breakout trading signal.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            BreakoutSignal with entry, stop, and take profit levels
        """
        # Calculate indicators
        analyzed = self.calculate_indicators(df)

        # Get current and previous bar values
        current = analyzed.iloc[-1]
        previous = analyzed.iloc[-2] if len(analyzed) > 1 else current

        price = float(current['Close'])
        high = float(current['High'])
        low = float(current['Low'])
        atr = float(current['ATR'])
        momentum = float(current['Momentum'])

        # Previous bar's channel (to detect breakout)
        channel_high = float(previous['Channel_High'])
        channel_low = float(previous['Channel_Low'])

        # Default values (no signal)
        signal = Signal.HOLD
        direction = 0
        entry_price = price
        stop_loss = price
        take_profit = price
        reason = "No breakout detected"

        # Check for LONG breakout
        if high > channel_high and momentum > self.momentum_threshold:
            signal = Signal.BUY
            direction = 1
            entry_price = channel_high + 0.01  # Entry just above breakout
            stop_loss = entry_price - (self.stop_atr_multiplier * atr)
            take_profit = entry_price + (self.tp_atr_multiplier * atr)
            reason = f"BREAKOUT LONG: Price {high:.2f} > Channel High {channel_high:.2f}, Momentum {momentum*100:+.1f}%"

        # Check for SHORT breakout
        elif low < channel_low and momentum < -self.momentum_threshold:
            signal = Signal.SELL
            direction = -1
            entry_price = channel_low - 0.01  # Entry just below breakout
            stop_loss = entry_price + (self.stop_atr_multiplier * atr)
            take_profit = entry_price - (self.tp_atr_multiplier * atr)
            reason = f"BREAKOUT SHORT: Price {low:.2f} < Channel Low {channel_low:.2f}, Momentum {momentum*100:+.1f}%"

        return BreakoutSignal(
            signal=signal,
            direction=direction,
            price=price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            momentum=momentum,
            channel_high=channel_high,
            channel_low=channel_low,
            timestamp=datetime.now(),
            reason=reason
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float
    ) -> int:
        """
        Calculate position size based on risk per trade.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            account_value: Total account value

        Returns:
            Number of shares to trade
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        max_risk = account_value * self.risk_per_trade
        position_size = int(max_risk / risk_per_share)

        # Limit to 50% of account value
        max_position_value = account_value * 0.5
        max_shares = int(max_position_value / entry_price)

        return min(position_size, max_shares)

    def get_channel_status(self, df: pd.DataFrame) -> dict:
        """Get current channel status for monitoring."""
        analyzed = self.calculate_indicators(df)
        current = analyzed.iloc[-1]

        price = float(current['Close'])
        channel_high = float(current['Channel_High'])
        channel_low = float(current['Channel_Low'])
        momentum = float(current['Momentum'])
        atr = float(current['ATR'])

        distance_to_high = ((channel_high - price) / price) * 100
        distance_to_low = ((price - channel_low) / price) * 100

        return {
            'price': price,
            'channel_high': channel_high,
            'channel_low': channel_low,
            'distance_to_high_pct': distance_to_high,
            'distance_to_low_pct': distance_to_low,
            'momentum_pct': momentum * 100,
            'atr': atr,
            'near_breakout_long': distance_to_high < 0.5 and momentum > 0,
            'near_breakout_short': distance_to_low < 0.5 and momentum < 0,
        }


@dataclass
class VolatilityBreakoutSignal:
    """Volatility Breakout trading signal."""
    signal: Signal
    direction: int  # 1 = LONG, -1 = SHORT, 0 = HOLD
    price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    volatility: float
    recent_high: float
    recent_low: float
    timestamp: datetime
    reason: str

    def __str__(self) -> str:
        dir_str = "LONG" if self.direction == 1 else "SHORT" if self.direction == -1 else "HOLD"
        return (
            f"{dir_str} @ ${self.price:.2f}\n"
            f"  Entry: ${self.entry_price:.2f} | Stop: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f}\n"
            f"  Volatility: {self.volatility*100:.2f}% | ATR: ${self.atr:.2f}\n"
            f"  Range: ${self.recent_low:.2f} - ${self.recent_high:.2f}\n"
            f"  Reason: {self.reason}"
        )


class VolatilityBreakoutSignalGenerator:
    """
    Generate trading signals based on Volatility Breakout strategy.

    Strategy (backtested +27.65% in 60 days vs -6.65% buy&hold):
    - Wait for LOW volatility (< 1.2%) = price compression
    - LONG: Price breaks above 7-bar high during compression
    - SHORT: Price breaks below 7-bar low during compression
    - Stop Loss: 1.5 × ATR
    - Take Profit: 3.5 × ATR

    The concept: Low volatility indicates price compression.
    When price breaks out after compression, moves are stronger.
    """

    def __init__(
        self,
        lookback_period: int = 7,
        volatility_threshold: float = 0.012,  # 1.2%
        volatility_period: int = 20,
        atr_period: int = 14,
        stop_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 3.5,
        risk_per_trade: float = 0.003,  # 0.3% risk per trade
    ):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.volatility_period = volatility_period
        self.atr_period = atr_period
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.risk_per_trade = risk_per_trade

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility breakout indicators."""
        result = df.copy()

        # Volatility (standard deviation / mean over N periods)
        result['Volatility'] = (
            df['Close'].rolling(window=self.volatility_period).std() /
            df['Close'].rolling(window=self.volatility_period).mean()
        )

        # Recent high/low (for breakout detection)
        result['Recent_High'] = df['High'].rolling(window=self.lookback_period).max().shift(1)
        result['Recent_Low'] = df['Low'].rolling(window=self.lookback_period).min().shift(1)

        # ATR for stop loss calculation
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['ATR'] = true_range.rolling(window=self.atr_period).mean()

        return result

    def generate_signal(self, df: pd.DataFrame) -> VolatilityBreakoutSignal:
        """
        Generate a volatility breakout trading signal.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            VolatilityBreakoutSignal with entry, stop, and take profit levels
        """
        # Calculate indicators
        analyzed = self.calculate_indicators(df)

        # Get current bar values
        current = analyzed.iloc[-1]

        price = float(current['Close'])
        high = float(current['High'])
        low = float(current['Low'])
        atr = float(current['ATR']) if not pd.isna(current['ATR']) else price * 0.01
        volatility = float(current['Volatility']) if not pd.isna(current['Volatility']) else 0.02
        recent_high = float(current['Recent_High']) if not pd.isna(current['Recent_High']) else high
        recent_low = float(current['Recent_Low']) if not pd.isna(current['Recent_Low']) else low

        # Default values (no signal)
        signal = Signal.HOLD
        direction = 0
        entry_price = price
        stop_loss = price
        take_profit = price
        reason = f"No breakout (Vol: {volatility*100:.2f}%, threshold: {self.volatility_threshold*100:.1f}%)"

        # Only trade when volatility is LOW (compression)
        if volatility < self.volatility_threshold:
            # Check for LONG breakout
            if price > recent_high:
                signal = Signal.BUY
                direction = 1
                entry_price = price
                stop_loss = entry_price - (self.stop_atr_multiplier * atr)
                take_profit = entry_price + (self.tp_atr_multiplier * atr)
                reason = f"VOL BREAKOUT LONG: Price ${price:.2f} > ${recent_high:.2f}, Vol {volatility*100:.2f}%"

            # Check for SHORT breakout
            elif price < recent_low:
                signal = Signal.SELL
                direction = -1
                entry_price = price
                stop_loss = entry_price + (self.stop_atr_multiplier * atr)
                take_profit = entry_price - (self.tp_atr_multiplier * atr)
                reason = f"VOL BREAKOUT SHORT: Price ${price:.2f} < ${recent_low:.2f}, Vol {volatility*100:.2f}%"

        return VolatilityBreakoutSignal(
            signal=signal,
            direction=direction,
            price=price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            volatility=volatility,
            recent_high=recent_high,
            recent_low=recent_low,
            timestamp=datetime.now(),
            reason=reason
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float
    ) -> int:
        """
        Calculate position size based on risk per trade.
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        max_risk = account_value * self.risk_per_trade
        position_size = int(max_risk / risk_per_share)

        # Limit to 95% of account value
        max_position_value = account_value * 0.95
        max_shares = int(max_position_value / entry_price)

        return min(position_size, max_shares)

    def get_status(self, df: pd.DataFrame) -> dict:
        """Get current volatility breakout status for monitoring."""
        analyzed = self.calculate_indicators(df)
        current = analyzed.iloc[-1]

        price = float(current['Close'])
        volatility = float(current['Volatility']) if not pd.isna(current['Volatility']) else 0
        recent_high = float(current['Recent_High']) if not pd.isna(current['Recent_High']) else price
        recent_low = float(current['Recent_Low']) if not pd.isna(current['Recent_Low']) else price
        atr = float(current['ATR']) if not pd.isna(current['ATR']) else 0

        distance_to_high = ((recent_high - price) / price) * 100
        distance_to_low = ((price - recent_low) / price) * 100

        is_compressed = volatility < self.volatility_threshold

        return {
            'price': price,
            'volatility_pct': volatility * 100,
            'volatility_threshold_pct': self.volatility_threshold * 100,
            'is_compressed': is_compressed,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'distance_to_high_pct': distance_to_high,
            'distance_to_low_pct': distance_to_low,
            'atr': atr,
            'ready_for_long': is_compressed and distance_to_high < 0.3,
            'ready_for_short': is_compressed and distance_to_low < 0.3,
        }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing BREAKOUT signal generation with NVDA...")
    nvda = yf.download("NVDA", period="5d", interval="5m", progress=False)

    # Handle multi-level columns
    if isinstance(nvda.columns, pd.MultiIndex):
        nvda.columns = nvda.columns.droplevel(1)

    # Test Breakout Generator
    breakout_gen = BreakoutSignalGenerator()
    signal = breakout_gen.generate_signal(nvda)

    print(f"\n=== BREAKOUT Signal ===")
    print(signal)

    print(f"\n=== Channel Status ===")
    status = breakout_gen.get_channel_status(nvda)
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Calculate position size example
    if signal.direction != 0:
        size = breakout_gen.calculate_position_size(
            signal.entry_price, signal.stop_loss, 100000
        )
        print(f"\n=== Position Size (100k account) ===")
        print(f"  Shares: {size}")
        print(f"  Value: ${size * signal.entry_price:,.2f}")
        print(f"  Risk: ${abs(signal.entry_price - signal.stop_loss) * size:,.2f}")
