"""
Backtesting de estrategias intradía optimizadas para NVDA.
Analiza la última semana y encuentra la mejor estrategia.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from itertools import product

console = Console()


def get_intraday_data(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Obtiene datos intradía de yfinance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df


def calculate_indicators(df: pd.DataFrame, rsi_period: int = 14,
                         ema_fast: int = 9, ema_slow: int = 21,
                         macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """Calcula indicadores técnicos."""
    df = df.copy()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # EMAs
    df['EMA_fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()

    # MACD
    ema_fast_macd = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow_macd = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast_macd - ema_slow_macd
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std * 2)
    df['BB_lower'] = df['BB_middle'] - (std * 2)

    # Volumen promedio
    df['Vol_avg'] = df['Volume'].rolling(window=20).mean()
    df['Vol_ratio'] = df['Volume'] / df['Vol_avg']

    return df


def strategy_rsi_macd(df: pd.DataFrame, rsi_oversold: int = 30, rsi_overbought: int = 70) -> pd.DataFrame:
    """Estrategia RSI + MACD: Compra cuando RSI < oversold y MACD cruza al alza."""
    df = df.copy()
    df['signal'] = 0

    # Señal de compra: RSI bajo + MACD cruza al alza
    buy_condition = (df['RSI'] < rsi_oversold) & (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)

    # Señal de venta: RSI alto o MACD cruza a la baja
    sell_condition = (df['RSI'] > rsi_overbought) | ((df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0))

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """Estrategia de cruce de EMAs."""
    df = df.copy()
    df['signal'] = 0

    # Compra cuando EMA rápida cruza por encima de EMA lenta
    buy_condition = (df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1))

    # Venta cuando EMA rápida cruza por debajo
    sell_condition = (df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1))

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_bollinger_rsi(df: pd.DataFrame, rsi_threshold: int = 40) -> pd.DataFrame:
    """Estrategia Bollinger Bands + RSI: Compra en banda inferior con RSI bajo."""
    df = df.copy()
    df['signal'] = 0

    # Compra cuando precio toca banda inferior y RSI < threshold
    buy_condition = (df['Close'] <= df['BB_lower']) & (df['RSI'] < rsi_threshold)

    # Venta cuando precio toca banda superior o RSI > 60
    sell_condition = (df['Close'] >= df['BB_upper']) | (df['RSI'] > 65)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_momentum_volume(df: pd.DataFrame, momentum_period: int = 5) -> pd.DataFrame:
    """Estrategia de momentum con confirmación de volumen."""
    df = df.copy()
    df['signal'] = 0
    df['momentum'] = df['Close'].pct_change(momentum_period)

    # Compra con momentum positivo y volumen alto
    buy_condition = (df['momentum'] > 0.005) & (df['Vol_ratio'] > 1.5) & (df['RSI'] < 65)

    # Venta con momentum negativo o RSI alto
    sell_condition = (df['momentum'] < -0.003) | (df['RSI'] > 75)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_scalping(df: pd.DataFrame) -> pd.DataFrame:
    """Estrategia de scalping: operaciones rápidas con pequeños movimientos."""
    df = df.copy()
    df['signal'] = 0

    # Compra cuando precio está cerca de BB_lower y RSI < 45
    buy_condition = (
        (df['Close'] < df['BB_middle']) &
        (df['RSI'] < 45) &
        (df['MACD_hist'] > df['MACD_hist'].shift(1))  # MACD mejorando
    )

    # Venta rápida: cualquier ganancia de 0.3% o pérdida de 0.2%
    sell_condition = (df['RSI'] > 55) | (df['Close'] > df['BB_middle'])

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                        oversold: int = 20, overbought: int = 80) -> pd.DataFrame:
    """Estrategia Stochastic Oscillator: compra en sobreventa, vende en sobrecompra."""
    df = df.copy()

    # Calcular Stochastic %K y %D
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

    df['signal'] = 0

    # Compra cuando %K cruza por encima de %D en zona de sobreventa
    buy_condition = (
        (df['stoch_k'] > df['stoch_d']) &
        (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) &
        (df['stoch_k'] < oversold + 20)
    )

    # Venta cuando %K cruza por debajo de %D en zona de sobrecompra
    sell_condition = (
        (df['stoch_k'] < df['stoch_d']) &
        (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) &
        (df['stoch_k'] > overbought - 20)
    ) | (df['stoch_k'] > overbought)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Estrategia VWAP: compra bajo VWAP, vende sobre VWAP."""
    df = df.copy()

    # Calcular VWAP
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['vwap_dist'] = (df['Close'] - df['vwap']) / df['vwap'] * 100

    df['signal'] = 0

    # Compra cuando precio está 0.3% bajo VWAP y RSI < 45
    buy_condition = (df['vwap_dist'] < -0.3) & (df['RSI'] < 45)

    # Venta cuando precio está 0.3% sobre VWAP o RSI > 60
    sell_condition = (df['vwap_dist'] > 0.3) | (df['RSI'] > 60)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_triple_ema(df: pd.DataFrame, ema1: int = 5, ema2: int = 13, ema3: int = 21) -> pd.DataFrame:
    """Estrategia Triple EMA: tendencia confirmada por 3 EMAs alineadas."""
    df = df.copy()

    df['ema1'] = df['Close'].ewm(span=ema1, adjust=False).mean()
    df['ema2'] = df['Close'].ewm(span=ema2, adjust=False).mean()
    df['ema3'] = df['Close'].ewm(span=ema3, adjust=False).mean()

    df['signal'] = 0

    # Compra cuando EMA1 > EMA2 > EMA3 (tendencia alcista)
    buy_condition = (
        (df['ema1'] > df['ema2']) &
        (df['ema2'] > df['ema3']) &
        (df['ema1'].shift(1) <= df['ema2'].shift(1))  # Cruce reciente
    )

    # Venta cuando EMA1 < EMA2 (tendencia se debilita)
    sell_condition = (df['ema1'] < df['ema2']) & (df['ema1'].shift(1) >= df['ema2'].shift(1))

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_mean_reversion(df: pd.DataFrame, std_mult: float = 1.5, lookback: int = 20) -> pd.DataFrame:
    """Estrategia Mean Reversion: compra cuando precio se desvía mucho de la media."""
    df = df.copy()

    df['sma'] = df['Close'].rolling(window=lookback).mean()
    df['std'] = df['Close'].rolling(window=lookback).std()
    df['z_score'] = (df['Close'] - df['sma']) / df['std']

    df['signal'] = 0

    # Compra cuando z-score < -std_mult (muy por debajo de la media)
    buy_condition = df['z_score'] < -std_mult

    # Venta cuando z-score > 0 (vuelve a la media) o > std_mult
    sell_condition = (df['z_score'] > 0) | (df['z_score'] > std_mult)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Estrategia Breakout: compra cuando rompe resistencia con volumen."""
    df = df.copy()

    df['resistance'] = df['High'].rolling(window=lookback).max()
    df['support'] = df['Low'].rolling(window=lookback).min()

    df['signal'] = 0

    # Compra en breakout alcista con volumen
    buy_condition = (
        (df['Close'] > df['resistance'].shift(1)) &
        (df['Vol_ratio'] > 1.3)
    )

    # Venta si pierde soporte o después de ganancia
    sell_condition = (df['Close'] < df['support'].shift(1)) | (df['RSI'] > 70)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_williams_r(df: pd.DataFrame, period: int = 14, oversold: int = -80, overbought: int = -20) -> pd.DataFrame:
    """Estrategia Williams %R: similar a Stochastic invertido."""
    df = df.copy()

    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min)

    df['signal'] = 0

    # Compra en sobreventa (< -80)
    buy_condition = (df['williams_r'] < oversold) & (df['williams_r'].shift(1) >= oversold)

    # Venta en sobrecompra (> -20)
    sell_condition = (df['williams_r'] > overbought) | (
        (df['williams_r'] > -50) & (df['williams_r'].shift(1) <= -50)
    )

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_adx_trend(df: pd.DataFrame, adx_period: int = 14, adx_threshold: int = 25) -> pd.DataFrame:
    """Estrategia ADX: opera solo cuando hay tendencia fuerte."""
    df = df.copy()

    # Calcular True Range y ADX
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=adx_period).mean()

    # +DI y -DI simplificados
    df['plus_dm'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0), 0
    )
    df['minus_dm'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
    )

    df['plus_di'] = 100 * df['plus_dm'].rolling(window=adx_period).mean() / df['atr']
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=adx_period).mean() / df['atr']

    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=adx_period).mean()

    df['signal'] = 0

    # Compra cuando ADX > threshold y +DI > -DI (tendencia alcista fuerte)
    buy_condition = (df['adx'] > adx_threshold) & (df['plus_di'] > df['minus_di']) & (df['plus_di'].shift(1) <= df['minus_di'].shift(1))

    # Venta cuando +DI cruza bajo -DI
    sell_condition = (df['plus_di'] < df['minus_di']) & (df['plus_di'].shift(1) >= df['minus_di'].shift(1))

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_rsi_divergence(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Estrategia RSI Divergence: busca divergencias entre precio y RSI."""
    df = df.copy()
    df['signal'] = 0

    # Divergencia alcista: precio hace mínimo más bajo, RSI hace mínimo más alto
    price_lower_low = df['Close'] < df['Close'].rolling(window=lookback).min().shift(1)
    rsi_higher_low = df['RSI'] > df['RSI'].rolling(window=lookback).min().shift(1)

    buy_condition = price_lower_low & rsi_higher_low & (df['RSI'] < 40)

    # Divergencia bajista: precio hace máximo más alto, RSI hace máximo más bajo
    price_higher_high = df['Close'] > df['Close'].rolling(window=lookback).max().shift(1)
    rsi_lower_high = df['RSI'] < df['RSI'].rolling(window=lookback).max().shift(1)

    sell_condition = (price_higher_high & rsi_lower_high) | (df['RSI'] > 65)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """Estrategia Keltner Channel: similar a Bollinger pero con ATR."""
    df = df.copy()

    df['kc_middle'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # ATR
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=ema_period).mean()

    df['kc_upper'] = df['kc_middle'] + (df['atr'] * atr_mult)
    df['kc_lower'] = df['kc_middle'] - (df['atr'] * atr_mult)

    df['signal'] = 0

    # Compra en banda inferior
    buy_condition = (df['Close'] <= df['kc_lower']) & (df['RSI'] < 40)

    # Venta en banda superior o media
    sell_condition = (df['Close'] >= df['kc_upper']) | ((df['Close'] > df['kc_middle']) & (df['RSI'] > 55))

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_macd_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Estrategia MACD Histogram: opera en cambios de dirección del histograma."""
    df = df.copy()
    df['signal'] = 0

    # Compra cuando histograma cambia de negativo a positivo y está mejorando
    buy_condition = (
        (df['MACD_hist'] > 0) &
        (df['MACD_hist'].shift(1) <= 0) &
        (df['RSI'] < 60)
    )

    # Venta cuando histograma cambia de positivo a negativo
    sell_condition = (
        (df['MACD_hist'] < 0) &
        (df['MACD_hist'].shift(1) >= 0)
    ) | (df['RSI'] > 70)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_double_bottom(df: pd.DataFrame, lookback: int = 30, tolerance: float = 0.02) -> pd.DataFrame:
    """Estrategia Double Bottom: detecta patrón de doble suelo."""
    df = df.copy()
    df['signal'] = 0

    df['local_min'] = df['Low'].rolling(window=lookback).min()
    df['prev_min'] = df['local_min'].shift(lookback // 2)

    # Detectar doble suelo: dos mínimos similares
    double_bottom = (
        (abs(df['Low'] - df['prev_min']) / df['prev_min'] < tolerance) &
        (df['Close'] > df['Low']) &
        (df['RSI'] < 45)
    )

    buy_condition = double_bottom
    sell_condition = (df['RSI'] > 60) | (df['Close'] < df['local_min'] * 0.98)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_gap_fade(df: pd.DataFrame, gap_threshold: float = 0.005) -> pd.DataFrame:
    """Estrategia Gap Fade: opera contra gaps intradía."""
    df = df.copy()
    df['signal'] = 0

    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Fade gap alcista (vender gap up)
    # Fade gap bajista (comprar gap down)
    buy_condition = (df['gap'] < -gap_threshold) & (df['RSI'] < 45)
    sell_condition = (df['gap'] > gap_threshold) | (df['RSI'] > 60)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def strategy_combo_rsi_stoch(df: pd.DataFrame) -> pd.DataFrame:
    """Estrategia Combo RSI + Stochastic: doble confirmación."""
    df = df.copy()

    # Calcular Stochastic
    k_period = 14
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)

    df['signal'] = 0

    # Compra cuando ambos están en sobreventa
    buy_condition = (df['RSI'] < 35) & (df['stoch_k'] < 25)

    # Venta cuando ambos están en sobrecompra
    sell_condition = (df['RSI'] > 65) | (df['stoch_k'] > 75)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


def backtest(df: pd.DataFrame, initial_capital: float = 10000,
             commission: float = 0.001, stop_loss: float = 0.02,
             take_profit: float = 0.03) -> dict:
    """Ejecuta backtest sobre las señales."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        signal = df['signal'].iloc[i]

        # Si tenemos posición, verificar stop loss / take profit
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price

            # Stop loss
            if pnl_pct <= -stop_loss:
                pnl = position * current_price * (1 - commission) - position * entry_price
                capital += position * current_price * (1 - commission)
                trades.append({
                    'type': 'STOP_LOSS',
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100
                })
                position = 0

            # Take profit
            elif pnl_pct >= take_profit:
                pnl = position * current_price * (1 - commission) - position * entry_price
                capital += position * current_price * (1 - commission)
                trades.append({
                    'type': 'TAKE_PROFIT',
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100
                })
                position = 0

            # Señal de venta
            elif signal == -1:
                pnl = position * current_price * (1 - commission) - position * entry_price
                capital += position * current_price * (1 - commission)
                trades.append({
                    'type': 'SIGNAL_SELL',
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100
                })
                position = 0

        # Señal de compra (sin posición)
        elif signal == 1 and position == 0:
            position = (capital * 0.95) / current_price  # 95% del capital
            entry_price = current_price
            capital -= position * entry_price * (1 + commission)

        # Actualizar equity curve
        current_equity = capital + (position * current_price if position > 0 else 0)
        equity_curve.append(current_equity)

    # Cerrar posición al final si existe
    if position > 0:
        final_price = df['Close'].iloc[-1]
        pnl = position * final_price * (1 - commission) - position * entry_price
        capital += position * final_price * (1 - commission)
        trades.append({
            'type': 'END_OF_PERIOD',
            'entry': entry_price,
            'exit': final_price,
            'pnl': pnl,
            'pnl_pct': ((final_price - entry_price) / entry_price) * 100
        })

    total_pnl = capital - initial_capital
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    return {
        'final_capital': capital,
        'total_pnl': total_pnl,
        'total_return_pct': (total_pnl / initial_capital) * 100,
        'num_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
        'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
        'trades': trades,
        'equity_curve': equity_curve
    }


def optimize_strategy(df_raw: pd.DataFrame, strategy_name: str) -> dict:
    """Optimiza parámetros de una estrategia."""
    best_result = None
    best_params = None

    if strategy_name == 'rsi_macd':
        for rsi_p, rsi_os, rsi_ob, sl, tp in product(
            [10, 14, 20],      # RSI period
            [25, 30, 35],      # RSI oversold
            [65, 70, 75],      # RSI overbought
            [0.015, 0.02, 0.025],  # stop loss
            [0.02, 0.03, 0.04]     # take profit
        ):
            df = calculate_indicators(df_raw, rsi_period=rsi_p)
            df = strategy_rsi_macd(df, rsi_oversold=rsi_os, rsi_overbought=rsi_ob)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'rsi_period': rsi_p, 'rsi_oversold': rsi_os,
                              'rsi_overbought': rsi_ob, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'ema_crossover':
        for ema_f, ema_s, sl, tp in product(
            [5, 9, 12],        # EMA fast
            [15, 21, 26],      # EMA slow
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            if ema_f >= ema_s:
                continue
            df = calculate_indicators(df_raw, ema_fast=ema_f, ema_slow=ema_s)
            df = strategy_ema_crossover(df)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'ema_fast': ema_f, 'ema_slow': ema_s,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'bollinger_rsi':
        for rsi_th, sl, tp in product(
            [35, 40, 45],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_bollinger_rsi(df, rsi_threshold=rsi_th)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'rsi_threshold': rsi_th, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'momentum_volume':
        for mom_p, sl, tp in product(
            [3, 5, 8],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_momentum_volume(df, momentum_period=mom_p)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'momentum_period': mom_p, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'scalping':
        for sl, tp in product(
            [0.01, 0.015, 0.02],
            [0.015, 0.02, 0.025]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_scalping(df)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'stochastic':
        for k_p, oversold, overbought, sl, tp in product(
            [10, 14, 20],
            [15, 20, 25],
            [75, 80, 85],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_stochastic(df, k_period=k_p, oversold=oversold, overbought=overbought)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'k_period': k_p, 'oversold': oversold, 'overbought': overbought,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'vwap':
        for sl, tp in product(
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_vwap(df)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'triple_ema':
        for ema1, ema2, ema3, sl, tp in product(
            [3, 5, 8],
            [10, 13, 15],
            [18, 21, 26],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            if ema1 >= ema2 or ema2 >= ema3:
                continue
            df = calculate_indicators(df_raw)
            df = strategy_triple_ema(df, ema1=ema1, ema2=ema2, ema3=ema3)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'ema1': ema1, 'ema2': ema2, 'ema3': ema3,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'mean_reversion':
        for std_mult, lookback, sl, tp in product(
            [1.0, 1.5, 2.0],
            [15, 20, 30],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_mean_reversion(df, std_mult=std_mult, lookback=lookback)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'std_mult': std_mult, 'lookback': lookback,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'breakout':
        for lookback, sl, tp in product(
            [15, 20, 30],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_breakout(df, lookback=lookback)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'lookback': lookback, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'williams_r':
        for period, oversold, overbought, sl, tp in product(
            [10, 14, 20],
            [-85, -80, -75],
            [-25, -20, -15],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_williams_r(df, period=period, oversold=oversold, overbought=overbought)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'period': period, 'oversold': oversold, 'overbought': overbought,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'adx_trend':
        for adx_period, adx_threshold, sl, tp in product(
            [10, 14, 20],
            [20, 25, 30],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_adx_trend(df, adx_period=adx_period, adx_threshold=adx_threshold)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'adx_period': adx_period, 'adx_threshold': adx_threshold,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'rsi_divergence':
        for lookback, sl, tp in product(
            [8, 10, 15],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_rsi_divergence(df, lookback=lookback)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'lookback': lookback, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'keltner_channel':
        for ema_period, atr_mult, sl, tp in product(
            [15, 20, 25],
            [1.5, 2.0, 2.5],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_keltner_channel(df, ema_period=ema_period, atr_mult=atr_mult)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'ema_period': ema_period, 'atr_mult': atr_mult,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'macd_histogram':
        for sl, tp in product(
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_macd_histogram(df)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'double_bottom':
        for lookback, tolerance, sl, tp in product(
            [20, 30, 40],
            [0.015, 0.02, 0.025],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_double_bottom(df, lookback=lookback, tolerance=tolerance)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'lookback': lookback, 'tolerance': tolerance,
                              'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'gap_fade':
        for gap_th, sl, tp in product(
            [0.003, 0.005, 0.007],
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_gap_fade(df, gap_threshold=gap_th)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'gap_threshold': gap_th, 'stop_loss': sl, 'take_profit': tp}

    elif strategy_name == 'combo_rsi_stoch':
        for sl, tp in product(
            [0.015, 0.02, 0.025],
            [0.02, 0.03, 0.04]
        ):
            df = calculate_indicators(df_raw)
            df = strategy_combo_rsi_stoch(df)
            result = backtest(df, stop_loss=sl, take_profit=tp)

            if best_result is None or result['total_pnl'] > best_result['total_pnl']:
                best_result = result
                best_params = {'stop_loss': sl, 'take_profit': tp}

    return {'result': best_result, 'params': best_params}


def run_analysis(symbol: str = "NVDA"):
    """Ejecuta análisis completo con todas las estrategias."""
    console.print(Panel.fit(f"[bold cyan]Análisis Intradía Optimizado - {symbol}[/bold cyan]"))

    console.print("\n[yellow]Descargando datos de los últimos 5 días...[/yellow]")
    df_raw = get_intraday_data(symbol, period="5d", interval="5m")

    if df_raw.empty:
        console.print("[red]Error: No se pudieron obtener datos[/red]")
        return

    console.print(f"[green]✓ {len(df_raw)} registros obtenidos[/green]")
    console.print(f"  Desde: {df_raw.index[0]}")
    console.print(f"  Hasta: {df_raw.index[-1]}")
    console.print(f"  Precio actual: ${df_raw['Close'].iloc[-1]:.2f}")

    strategies = [
        'rsi_macd', 'ema_crossover', 'bollinger_rsi', 'momentum_volume', 'scalping',
        'stochastic', 'vwap', 'triple_ema', 'mean_reversion', 'breakout',
        'williams_r', 'adx_trend', 'rsi_divergence', 'keltner_channel',
        'macd_histogram', 'double_bottom', 'gap_fade', 'combo_rsi_stoch'
    ]
    results = {}

    console.print("\n[yellow]Optimizando estrategias...[/yellow]")

    for strategy in strategies:
        console.print(f"  Optimizando {strategy}...", end=" ")
        results[strategy] = optimize_strategy(df_raw, strategy)
        pnl = results[strategy]['result']['total_pnl']
        color = "green" if pnl > 0 else "red"
        console.print(f"[{color}]${pnl:+.2f}[/{color}]")

    # Mostrar tabla de resultados
    console.print("\n")
    table = Table(title="Resultados del Backtesting (Capital inicial: $10,000)")
    table.add_column("Estrategia", style="cyan")
    table.add_column("P&L", justify="right")
    table.add_column("Retorno %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Parámetros Óptimos", style="dim")

    for name, data in sorted(results.items(), key=lambda x: x[1]['result']['total_pnl'], reverse=True):
        r = data['result']
        pnl_color = "green" if r['total_pnl'] > 0 else "red"

        params_str = ", ".join([f"{k}={v}" for k, v in data['params'].items()])

        table.add_row(
            name,
            f"[{pnl_color}]${r['total_pnl']:+.2f}[/{pnl_color}]",
            f"[{pnl_color}]{r['total_return_pct']:+.2f}%[/{pnl_color}]",
            str(r['num_trades']),
            f"{r['win_rate']:.1f}%",
            params_str[:50] + "..." if len(params_str) > 50 else params_str
        )

    console.print(table)

    # Mejor estrategia
    best_strategy = max(results.items(), key=lambda x: x[1]['result']['total_pnl'])
    best_name = best_strategy[0]
    best_data = best_strategy[1]

    console.print("\n")
    console.print(Panel.fit(f"""[bold green]MEJOR ESTRATEGIA: {best_name.upper()}[/bold green]

💰 Ganancia: ${best_data['result']['total_pnl']:+.2f}
📈 Retorno: {best_data['result']['total_return_pct']:+.2f}%
🎯 Trades: {best_data['result']['num_trades']}
✅ Win Rate: {best_data['result']['win_rate']:.1f}%
💵 Ganancia promedio: ${best_data['result']['avg_win']:.2f}
📉 Pérdida promedio: ${best_data['result']['avg_loss']:.2f}

Parámetros óptimos:
{chr(10).join([f'  • {k}: {v}' for k, v in best_data['params'].items()])}
"""))

    # Mostrar trades de la mejor estrategia
    if best_data['result']['trades']:
        console.print("\n[bold]Últimos trades de la mejor estrategia:[/bold]")
        trades_table = Table()
        trades_table.add_column("Tipo", style="cyan")
        trades_table.add_column("Entrada", justify="right")
        trades_table.add_column("Salida", justify="right")
        trades_table.add_column("P&L", justify="right")
        trades_table.add_column("P&L %", justify="right")

        for trade in best_data['result']['trades'][-10:]:
            pnl_color = "green" if trade['pnl'] > 0 else "red"
            trades_table.add_row(
                trade['type'],
                f"${trade['entry']:.2f}",
                f"${trade['exit']:.2f}",
                f"[{pnl_color}]${trade['pnl']:+.2f}[/{pnl_color}]",
                f"[{pnl_color}]{trade['pnl_pct']:+.2f}%[/{pnl_color}]"
            )

        console.print(trades_table)

    # Resumen diario
    console.print("\n[bold]Proyección diaria (basada en datos de 5 días):[/bold]")
    daily_pnl = best_data['result']['total_pnl'] / 5
    console.print(f"  Ganancia diaria promedio: [green]${daily_pnl:+.2f}[/green]")
    console.print(f"  Ganancia semanal proyectada: [green]${daily_pnl * 5:+.2f}[/green]")
    console.print(f"  Ganancia mensual proyectada: [green]${daily_pnl * 22:+.2f}[/green]")

    return results


if __name__ == "__main__":
    run_analysis("NVDA")
