# Market Analyzer

Sistema de análisis de acciones en tiempo real con valuations, fundamentals y análisis técnico.

## Características

- **Datos en tiempo real**: Precios y volumen de acciones
- **Análisis Fundamental**: P/E, P/B, ROE, deuda, flujo de caja
- **Valuaciones**: DCF, comparables, múltiplos
- **Análisis Técnico**: RSI, MACD, medias móviles, soportes/resistencias

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python -m market_analyzer --symbol AAPL
```

## Estructura del proyecto

```
market_analyzer/
├── core/           # Lógica principal
├── data/           # Obtención de datos (APIs)
├── fundamentals/   # Análisis fundamental
├── technicals/     # Análisis técnico
├── valuations/     # Modelos de valuación
└── utils/          # Utilidades
```

## Licencia

MIT
