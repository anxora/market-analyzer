"""
NVDA Intraday Trading Bot

Automated trading based on RSI, MACD, and VWAP indicators.
Supports Interactive Brokers API integration.

Usage:
    # Paper trading simulation
    python -m market_analyzer.trading_bot.bot --symbol NVDA --paper

    # With IB Gateway (paper)
    python -m market_analyzer.trading_bot.bot --symbol NVDA --ib-paper

    # Live trading
    python -m market_analyzer.trading_bot.bot --symbol NVDA --live

Risk Management:
    - Default max risk: 0.5% per trade
    - Stop loss: 0.5% below entry
    - Take profit: 1.0% above entry (2:1 risk/reward)
"""

from .indicators import (
    TechnicalAnalyzer,
    IndicatorValues,
    calculate_rsi,
    calculate_macd,
    calculate_vwap,
    calculate_atr,
    calculate_ema,
    calculate_sma,
    calculate_bollinger_bands,
)

from .signals import (
    SignalGenerator,
    Signal,
    SignalStrength,
    TradeSignal,
)

from .broker import (
    IBBroker,
    SimulatedBroker,
    OrderResult,
    OrderType,
)

from .bot import TradingBot

__all__ = [
    # Indicators
    'TechnicalAnalyzer',
    'IndicatorValues',
    'calculate_rsi',
    'calculate_macd',
    'calculate_vwap',
    'calculate_atr',
    'calculate_ema',
    'calculate_sma',
    'calculate_bollinger_bands',
    # Signals
    'SignalGenerator',
    'Signal',
    'SignalStrength',
    'TradeSignal',
    # Broker
    'IBBroker',
    'SimulatedBroker',
    'OrderResult',
    'OrderType',
    # Bot
    'TradingBot',
]
