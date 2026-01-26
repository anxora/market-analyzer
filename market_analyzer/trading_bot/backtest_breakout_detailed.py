"""
Backtest detallado de la estrategia Breakout para mostrar exactamente
cómo se llega al ~24% de retorno.

Ejecutar: python -m market_analyzer.trading_bot.backtest_breakout_detailed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
import yfinance as yf

from .indicators import TechnicalAnalyzer


@dataclass
class Trade:
    """Registro de un trade individual."""
    num: int
    entry_time: datetime
    exit_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    momentum_at_entry: float
    atr_at_entry: float
    channel_high: float
    channel_low: float


def run_detailed_backtest(
    symbol: str = "NVDA",
    period: str = "60d",
    interval: str = "5m",
    initial_capital: float = 100000,
    channel_period: int = 20,
    momentum_period: int = 10,
    momentum_threshold: float = 0.01,  # 1%
    stop_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    risk_per_trade: float = 0.003,  # 0.3%
    commission_per_share: float = 0.005,
):
    """
    Ejecuta backtest detallado mostrando cada trade.
    """
    print("=" * 70)
    print("BACKTEST DETALLADO - ESTRATEGIA DONCHIAN CHANNEL BREAKOUT")
    print("=" * 70)

    # Parámetros
    print(f"\n📊 PARÁMETROS:")
    print(f"   Símbolo: {symbol}")
    print(f"   Período: {period}")
    print(f"   Intervalo: {interval}")
    print(f"   Capital inicial: ${initial_capital:,.2f}")
    print(f"   Canal Donchian: {channel_period} barras")
    print(f"   Momentum threshold: {momentum_threshold*100:.1f}%")
    print(f"   Stop Loss: {stop_atr_mult}× ATR")
    print(f"   Take Profit: {tp_atr_mult}× ATR")
    print(f"   Riesgo por trade: {risk_per_trade*100:.1f}%")
    print(f"   Comisión: ${commission_per_share}/acción")

    # Descargar datos
    print(f"\n📥 Descargando datos de {symbol}...")
    data = yf.download(symbol, period=period, interval=interval, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    print(f"   {len(data)} barras descargadas")
    print(f"   Desde: {data.index[0]}")
    print(f"   Hasta: {data.index[-1]}")

    # Calcular indicadores
    print(f"\n📈 Calculando indicadores...")

    # Donchian Channel (shifted by 1 to avoid look-ahead bias)
    data['channel_high'] = data['High'].rolling(window=channel_period).max().shift(1)
    data['channel_low'] = data['Low'].rolling(window=channel_period).min().shift(1)

    # Momentum
    data['momentum'] = data['Close'].pct_change(periods=momentum_period)

    # ATR
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(window=14).mean()

    # Simulación
    print(f"\n🔄 EJECUTANDO SIMULACIÓN...")
    print("-" * 70)

    capital = initial_capital
    position = None
    trades: List[Trade] = []
    trade_num = 0
    equity_curve = []

    min_bars = max(channel_period, momentum_period, 14) + 5

    for i in range(min_bars, len(data)):
        row = data.iloc[i]
        timestamp = data.index[i]
        price = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])

        channel_high = float(row['channel_high'])
        channel_low = float(row['channel_low'])
        momentum = float(row['momentum'])
        atr = float(row['atr'])

        if pd.isna(channel_high) or pd.isna(atr) or pd.isna(momentum):
            continue

        # Verificar salida si hay posición
        if position is not None:
            exit_price = None
            exit_reason = None

            if position['direction'] == "LONG":
                if low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "STOP LOSS"
                elif high >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TAKE PROFIT"
            else:  # SHORT
                if high >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "STOP LOSS"
                elif low <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "TAKE PROFIT"

            if exit_price is not None:
                # Cerrar posición
                if position['direction'] == "LONG":
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']

                # Restar comisiones
                commission = position['quantity'] * commission_per_share * 2
                pnl -= commission

                pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) * 100
                capital += pnl

                trade = Trade(
                    num=trade_num,
                    entry_time=position['entry_time'],
                    exit_time=timestamp,
                    direction=position['direction'],
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    stop_loss=position['stop_loss'],
                    take_profit=position['take_profit'],
                    quantity=position['quantity'],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    momentum_at_entry=position['momentum'],
                    atr_at_entry=position['atr'],
                    channel_high=position['channel_high'],
                    channel_low=position['channel_low'],
                )
                trades.append(trade)
                position = None

        # Verificar entrada si no hay posición
        if position is None:
            signal_direction = None

            # LONG: Precio rompe máximo del canal + momentum positivo
            if price > channel_high and momentum > momentum_threshold:
                signal_direction = "LONG"
                entry_price = price
                stop_loss = entry_price - (atr * stop_atr_mult)
                take_profit = entry_price + (atr * tp_atr_mult)

            # SHORT: Precio rompe mínimo del canal + momentum negativo
            elif price < channel_low and momentum < -momentum_threshold:
                signal_direction = "SHORT"
                entry_price = price
                stop_loss = entry_price + (atr * stop_atr_mult)
                take_profit = entry_price - (atr * tp_atr_mult)

            if signal_direction:
                # Calcular tamaño de posición basado en riesgo
                risk_amount = capital * risk_per_trade
                risk_per_share = abs(entry_price - stop_loss)
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                # Limitar a lo que podemos permitirnos
                max_affordable = int((capital * 0.95) / entry_price)
                quantity = min(quantity, max_affordable)

                if quantity > 0:
                    trade_num += 1
                    position = {
                        'direction': signal_direction,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'quantity': quantity,
                        'entry_time': timestamp,
                        'momentum': momentum,
                        'atr': atr,
                        'channel_high': channel_high,
                        'channel_low': channel_low,
                    }

        # Actualizar equity curve
        if position is not None:
            if position['direction'] == "LONG":
                unrealized = (price - position['entry_price']) * position['quantity']
            else:
                unrealized = (position['entry_price'] - price) * position['quantity']
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)

    # Cerrar posición final si existe
    if position is not None:
        final_price = float(data['Close'].iloc[-1])
        if position['direction'] == "LONG":
            pnl = (final_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - final_price) * position['quantity']
        commission = position['quantity'] * commission_per_share * 2
        pnl -= commission
        capital += pnl

        trade = Trade(
            num=trade_num,
            entry_time=position['entry_time'],
            exit_time=data.index[-1],
            direction=position['direction'],
            entry_price=position['entry_price'],
            exit_price=final_price,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            quantity=position['quantity'],
            pnl=pnl,
            pnl_pct=(pnl / (position['entry_price'] * position['quantity'])) * 100,
            exit_reason="FIN BACKTEST",
            momentum_at_entry=position['momentum'],
            atr_at_entry=position['atr'],
            channel_high=position['channel_high'],
            channel_low=position['channel_low'],
        )
        trades.append(trade)

    # Mostrar trades detallados
    print(f"\n📋 LISTA DETALLADA DE TRADES ({len(trades)} trades):")
    print("=" * 120)

    total_pnl = 0
    running_capital = initial_capital

    for t in trades:
        emoji = "✅" if t.pnl > 0 else "❌"
        running_capital += t.pnl
        total_pnl += t.pnl

        print(f"\n{emoji} TRADE #{t.num} - {t.direction}")
        print(f"   Entrada: {t.entry_time.strftime('%Y-%m-%d %H:%M')} @ ${t.entry_price:.2f}")
        print(f"   Salida:  {t.exit_time.strftime('%Y-%m-%d %H:%M')} @ ${t.exit_price:.2f} ({t.exit_reason})")
        print(f"   Cantidad: {t.quantity} acciones")
        print(f"   Stop Loss: ${t.stop_loss:.2f} | Take Profit: ${t.take_profit:.2f}")
        print(f"   Canal al entrar: ${t.channel_low:.2f} - ${t.channel_high:.2f}")
        print(f"   Momentum: {t.momentum_at_entry*100:+.2f}% | ATR: ${t.atr_at_entry:.2f}")
        print(f"   P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.2f}%)")
        print(f"   Capital acumulado: ${running_capital:,.2f}")

    # Estadísticas
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 70)

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]

    total_return = capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0

    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = gross_profit / len(winning_trades) if winning_trades else 0
    avg_loss = gross_loss / len(losing_trades) if losing_trades else 0

    # Drawdown
    equity = pd.Series(equity_curve)
    rolling_max = equity.expanding().max()
    drawdown = equity - rolling_max
    max_dd = abs(drawdown.min())
    max_dd_pct = (max_dd / rolling_max[drawdown.idxmin()]) * 100 if len(equity) > 0 and max_dd > 0 else 0

    print(f"\n💰 RENDIMIENTO:")
    print(f"   Capital Inicial:     ${initial_capital:,.2f}")
    print(f"   Capital Final:       ${capital:,.2f}")
    print(f"   Retorno Total:       ${total_return:+,.2f} ({total_return_pct:+.2f}%)")

    print(f"\n📈 TRADES:")
    print(f"   Total Trades:        {len(trades)}")
    print(f"   Trades Ganadores:    {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"   Trades Perdedores:   {len(losing_trades)}")

    print(f"\n💵 GANANCIAS/PÉRDIDAS:")
    print(f"   Profit Bruto:        ${gross_profit:+,.2f}")
    print(f"   Loss Bruto:          ${-gross_loss:,.2f}")
    print(f"   Profit Factor:       {profit_factor:.2f}")
    print(f"   Promedio Ganancia:   ${avg_win:+,.2f}")
    print(f"   Promedio Pérdida:    ${-avg_loss:,.2f}")

    if winning_trades:
        print(f"   Mayor Ganancia:      ${max(t.pnl for t in winning_trades):+,.2f}")
    if losing_trades:
        print(f"   Mayor Pérdida:       ${min(t.pnl for t in losing_trades):+,.2f}")

    print(f"\n⚠️ RIESGO:")
    print(f"   Max Drawdown:        ${max_dd:,.2f} ({max_dd_pct:.2f}%)")

    # Desglose por tipo de salida
    print(f"\n🎯 DESGLOSE POR TIPO DE SALIDA:")
    tp_trades = [t for t in trades if t.exit_reason == "TAKE PROFIT"]
    sl_trades = [t for t in trades if t.exit_reason == "STOP LOSS"]
    other_trades = [t for t in trades if t.exit_reason not in ["TAKE PROFIT", "STOP LOSS"]]

    print(f"   Take Profit:  {len(tp_trades)} trades (${sum(t.pnl for t in tp_trades):+,.2f})")
    print(f"   Stop Loss:    {len(sl_trades)} trades (${sum(t.pnl for t in sl_trades):+,.2f})")
    if other_trades:
        print(f"   Otros:        {len(other_trades)} trades (${sum(t.pnl for t in other_trades):+,.2f})")

    # Desglose por dirección
    print(f"\n📊 DESGLOSE POR DIRECCIÓN:")
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]

    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)

    print(f"   LONG:   {len(long_trades)} trades (${long_pnl:+,.2f})")
    print(f"   SHORT:  {len(short_trades)} trades (${short_pnl:+,.2f})")

    print("\n" + "=" * 70)
    print("FIN DEL BACKTEST")
    print("=" * 70)

    return {
        'trades': trades,
        'capital': capital,
        'return_pct': total_return_pct,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd_pct,
    }


if __name__ == "__main__":
    run_detailed_backtest()
