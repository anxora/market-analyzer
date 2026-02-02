"""
Análisis de compatibilidad entre las top 6 estrategias.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from market_analyzer.backtest_intraday import (
    get_intraday_data, calculate_indicators,
    strategy_bollinger_rsi, strategy_stochastic, strategy_mean_reversion,
    strategy_adx_trend, strategy_rsi_divergence, strategy_keltner_channel,
    backtest
)

console = Console()


def analyze_compatibility():
    """Analiza si las estrategias generan señales compatibles."""

    console.print(Panel.fit("[bold cyan]Análisis de Compatibilidad - Top 6 Estrategias[/bold cyan]"))

    # Obtener datos
    console.print("\n[yellow]Descargando datos de NVDA...[/yellow]")
    df_raw = get_intraday_data("NVDA", period="5d", interval="5m")
    df = calculate_indicators(df_raw)

    console.print(f"[green]✓ {len(df)} registros[/green]\n")

    # Aplicar cada estrategia con sus parámetros óptimos
    strategies = {
        'Keltner Channel': strategy_keltner_channel(df.copy(), ema_period=15, atr_mult=2.0),
        'Mean Reversion': strategy_mean_reversion(df.copy(), std_mult=2.0, lookback=20),
        'ADX Trend': strategy_adx_trend(df.copy(), adx_period=14, adx_threshold=25),
        'Bollinger RSI': strategy_bollinger_rsi(df.copy(), rsi_threshold=45),
        'Stochastic': strategy_stochastic(df.copy(), k_period=20, oversold=20, overbought=85),
        'RSI Divergence': strategy_rsi_divergence(df.copy(), lookback=15),
    }

    # Crear DataFrame con todas las señales
    signals_df = pd.DataFrame(index=df.index)
    for name, strat_df in strategies.items():
        signals_df[name] = strat_df['signal']

    # Análisis de señales
    console.print("[bold]1. RESUMEN DE SEÑALES POR ESTRATEGIA[/bold]\n")

    summary_table = Table()
    summary_table.add_column("Estrategia", style="cyan")
    summary_table.add_column("Compras", justify="right")
    summary_table.add_column("Ventas", justify="right")
    summary_table.add_column("Total Señales", justify="right")

    for name in strategies.keys():
        buys = (signals_df[name] == 1).sum()
        sells = (signals_df[name] == -1).sum()
        summary_table.add_row(name, str(buys), str(sells), str(buys + sells))

    console.print(summary_table)

    # Análisis de coincidencias
    console.print("\n[bold]2. MATRIZ DE COINCIDENCIA DE SEÑALES[/bold]")
    console.print("[dim]% de veces que dos estrategias generan la misma señal (cuando ambas tienen señal)[/dim]\n")

    strat_names = list(strategies.keys())

    # Calcular coincidencias
    coincidence_matrix = {}
    for i, name1 in enumerate(strat_names):
        coincidence_matrix[name1] = {}
        for j, name2 in enumerate(strat_names):
            if i == j:
                coincidence_matrix[name1][name2] = 100.0
            else:
                # Momentos donde ambas tienen señal (no 0)
                both_signal = (signals_df[name1] != 0) & (signals_df[name2] != 0)
                if both_signal.sum() > 0:
                    same_signal = (signals_df[name1] == signals_df[name2]) & both_signal
                    coincidence_matrix[name1][name2] = (same_signal.sum() / both_signal.sum()) * 100
                else:
                    coincidence_matrix[name1][name2] = None

    # Mostrar matriz
    matrix_table = Table()
    matrix_table.add_column("", style="cyan")
    for name in strat_names:
        matrix_table.add_column(name[:8], justify="center")

    for name1 in strat_names:
        row = [name1[:15]]
        for name2 in strat_names:
            val = coincidence_matrix[name1][name2]
            if val is None:
                row.append("-")
            elif val == 100.0 and name1 == name2:
                row.append("[dim]---[/dim]")
            elif val >= 70:
                row.append(f"[green]{val:.0f}%[/green]")
            elif val >= 40:
                row.append(f"[yellow]{val:.0f}%[/yellow]")
            else:
                row.append(f"[red]{val:.0f}%[/red]")
        matrix_table.add_row(*row)

    console.print(matrix_table)
    console.print("\n[dim]Verde: >70% coincidencia | Amarillo: 40-70% | Rojo: <40%[/dim]")

    # Análisis de conflictos
    console.print("\n[bold]3. ANÁLISIS DE CONFLICTOS[/bold]")
    console.print("[dim]Momentos donde una estrategia dice COMPRAR y otra dice VENDER[/dim]\n")

    conflicts = {}
    for i, name1 in enumerate(strat_names):
        for j, name2 in enumerate(strat_names):
            if i < j:
                # Conflicto: una dice comprar, otra dice vender
                conflict = ((signals_df[name1] == 1) & (signals_df[name2] == -1)) | \
                          ((signals_df[name1] == -1) & (signals_df[name2] == 1))
                conflicts[(name1, name2)] = conflict.sum()

    conflict_table = Table()
    conflict_table.add_column("Par de Estrategias", style="cyan")
    conflict_table.add_column("Conflictos", justify="right")
    conflict_table.add_column("Compatibilidad", justify="center")

    for (name1, name2), count in sorted(conflicts.items(), key=lambda x: x[1]):
        if count == 0:
            compat = "[green]PERFECTA[/green]"
        elif count <= 3:
            compat = "[green]ALTA[/green]"
        elif count <= 7:
            compat = "[yellow]MEDIA[/yellow]"
        else:
            compat = "[red]BAJA[/red]"
        conflict_table.add_row(f"{name1[:12]} + {name2[:12]}", str(count), compat)

    console.print(conflict_table)

    # Clasificación por tipo de estrategia
    console.print("\n[bold]4. CLASIFICACIÓN POR TIPO[/bold]\n")

    classification = """
    [cyan]MEAN REVERSION (Reversión a la media):[/cyan]
      • Keltner Channel
      • Mean Reversion
      • Bollinger RSI
      → [green]COMPATIBLES entre sí[/green] - Misma filosofía

    [cyan]MOMENTUM/OSCILADORES:[/cyan]
      • Stochastic
      • RSI Divergence
      → [green]COMPATIBLES entre sí[/green] - Buscan sobreventa/sobrecompra

    [cyan]TENDENCIA:[/cyan]
      • ADX Trend
      → [yellow]PARCIALMENTE COMPATIBLE[/yellow] - Opera solo con tendencia fuerte
    """
    console.print(classification)

    # Estrategia combinada
    console.print("\n[bold]5. ESTRATEGIA COMBINADA (CONSENSO)[/bold]")
    console.print("[dim]Compra solo cuando 3+ estrategias coinciden[/dim]\n")

    # Contar votos
    signals_df['buy_votes'] = (signals_df[strat_names] == 1).sum(axis=1)
    signals_df['sell_votes'] = (signals_df[strat_names] == -1).sum(axis=1)

    # Señal combinada: requiere 3+ votos
    signals_df['combined_signal'] = 0
    signals_df.loc[signals_df['buy_votes'] >= 3, 'combined_signal'] = 1
    signals_df.loc[signals_df['sell_votes'] >= 3, 'combined_signal'] = -1

    # Backtest de estrategia combinada
    df_combined = df.copy()
    df_combined['signal'] = signals_df['combined_signal']

    result_combined = backtest(df_combined, stop_loss=0.02, take_profit=0.025)

    console.print(f"  Señales de compra con consenso: {(signals_df['combined_signal'] == 1).sum()}")
    console.print(f"  Señales de venta con consenso: {(signals_df['combined_signal'] == -1).sum()}")
    console.print(f"\n  [bold]Resultado del backtesting combinado:[/bold]")

    pnl_color = "green" if result_combined['total_pnl'] > 0 else "red"
    console.print(f"    P&L: [{pnl_color}]${result_combined['total_pnl']:+.2f}[/{pnl_color}]")
    console.print(f"    Trades: {result_combined['num_trades']}")
    console.print(f"    Win Rate: {result_combined['win_rate']:.1f}%")

    # Mostrar momentos de consenso
    console.print("\n[bold]6. MOMENTOS DE MAYOR CONSENSO (3+ estrategias)[/bold]\n")

    consensus_buys = signals_df[signals_df['buy_votes'] >= 3].copy()
    if len(consensus_buys) > 0:
        console.print("[green]Señales de COMPRA con consenso:[/green]")
        for idx in consensus_buys.index[-5:]:
            votes = consensus_buys.loc[idx, 'buy_votes']
            price = df.loc[idx, 'Close']
            console.print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - ${price:.2f} ({votes} votos)")

    consensus_sells = signals_df[signals_df['sell_votes'] >= 3].copy()
    if len(consensus_sells) > 0:
        console.print("\n[red]Señales de VENTA con consenso:[/red]")
        for idx in consensus_sells.index[-5:]:
            votes = consensus_sells.loc[idx, 'sell_votes']
            price = df.loc[idx, 'Close']
            console.print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - ${price:.2f} ({votes} votos)")

    # Recomendación final
    console.print("\n")
    console.print(Panel.fit("""[bold green]RECOMENDACIÓN DE COMBINACIÓN[/bold green]

Las estrategias TOP 6 se dividen en dos grupos compatibles:

[cyan]GRUPO A - Reversión a la media:[/cyan]
  Keltner + Mean Reversion + Bollinger RSI
  → Usar cuando NVDA está en RANGO LATERAL

[cyan]GRUPO B - Momentum:[/cyan]
  Stochastic + RSI Divergence
  → Confirman entradas del Grupo A

[cyan]GRUPO C - Tendencia:[/cyan]
  ADX Trend
  → Usar cuando hay TENDENCIA CLARA (ADX > 25)

[yellow]ESTRATEGIA ÓPTIMA:[/yellow]
1. Si ADX < 25 (sin tendencia): Usar Grupo A + B
2. Si ADX > 25 (con tendencia): Usar ADX Trend
3. Entrar solo con 2+ confirmaciones del mismo grupo
"""))

    return signals_df


if __name__ == "__main__":
    analyze_compatibility()
