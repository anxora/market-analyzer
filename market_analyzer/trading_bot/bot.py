"""
NVDA Intraday Trading Bot

Trades NVDA based on RSI, MACD, and VWAP signals with strict risk management.
Maximum risk per trade: 0.5% of account value.

Usage:
    # Paper trading (simulated)
    python -m market_analyzer.trading_bot.bot --symbol NVDA --paper

    # Paper trading with IB Gateway
    python -m market_analyzer.trading_bot.bot --symbol NVDA --ib-paper

    # Live trading (requires IB Gateway live connection)
    python -m market_analyzer.trading_bot.bot --symbol NVDA --live
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf

from .indicators import TechnicalAnalyzer
from .signals import SignalGenerator, Signal, TradeSignal
from .broker import IBBroker, SimulatedBroker, OrderResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Intraday trading bot for NVDA using technical indicators.

    Risk Management:
    - Maximum 0.5% loss per trade (configurable)
    - Stop loss set at entry price - 0.5%
    - Position size calculated to risk exactly max_risk_pct
    - Only one position at a time
    """

    def __init__(
        self,
        symbol: str = "NVDA",
        max_risk_pct: float = 0.5,
        check_interval_seconds: int = 60,
        use_ib: bool = False,
        ib_port: int = 4002,
        paper_trading: bool = True,
        initial_capital: float = 100000
    ):
        self.symbol = symbol
        self.max_risk_pct = max_risk_pct
        self.check_interval = check_interval_seconds
        self.paper_trading = paper_trading

        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            rsi_oversold=30,
            rsi_overbought=70,
            require_confluence=2
        )

        # Initialize broker
        if use_ib:
            self.broker = IBBroker(
                port=ib_port,
                max_risk_pct=max_risk_pct,
                paper_trading=paper_trading
            )
        else:
            self.broker = SimulatedBroker(
                initial_capital=initial_capital,
                max_risk_pct=max_risk_pct
            )

        self.running = False
        self.last_signal: Optional[TradeSignal] = None
        self.trade_log = []

    def get_intraday_data(self, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data for analysis."""
        try:
            data = yf.download(
                self.symbol,
                period=period,
                interval=interval,
                progress=False
            )
            if data.empty:
                logger.warning(f"No data received for {self.symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_stop_loss(self, entry_price: float, action: str) -> float:
        """
        Calculate stop loss price for 0.5% maximum risk.

        Args:
            entry_price: Entry price
            action: "BUY" or "SELL"

        Returns:
            Stop loss price
        """
        if action == "BUY":
            # Stop loss 0.5% below entry for long positions
            return entry_price * (1 - self.max_risk_pct / 100)
        else:
            # Stop loss 0.5% above entry for short positions
            return entry_price * (1 + self.max_risk_pct / 100)

    def calculate_take_profit(self, entry_price: float, action: str, risk_reward: float = 2.0) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            action: "BUY" or "SELL"
            risk_reward: Risk/reward ratio (default 2:1)

        Returns:
            Take profit price
        """
        profit_pct = self.max_risk_pct * risk_reward
        if action == "BUY":
            return entry_price * (1 + profit_pct / 100)
        else:
            return entry_price * (1 - profit_pct / 100)

    def should_trade(self) -> bool:
        """Check if within trading hours (9:30 AM - 4:00 PM ET)."""
        now = datetime.now()
        # Simple check - in production, use proper timezone handling
        hour = now.hour
        minute = now.minute

        # Trading hours: 9:30 AM - 4:00 PM
        if hour < 9 or (hour == 9 and minute < 30):
            return False
        if hour >= 16:
            return False
        return True

    def execute_trade(self, signal: TradeSignal) -> Optional[OrderResult]:
        """
        Execute a trade based on the signal.

        Args:
            signal: TradeSignal with entry details

        Returns:
            OrderResult or None
        """
        if signal.signal == Signal.HOLD:
            return None

        # Check for existing position
        position = self.broker.get_position(self.symbol)
        if position and position['quantity'] != 0:
            logger.info(f"Already have position in {self.symbol}: {position['quantity']} shares")

            # Check if signal is opposite (exit signal)
            if (signal.signal == Signal.SELL and position['quantity'] > 0) or \
               (signal.signal == Signal.BUY and position['quantity'] < 0):
                logger.info("Opposite signal received - closing position")
                if isinstance(self.broker, SimulatedBroker):
                    return self.broker.place_market_order(
                        self.symbol,
                        "SELL" if position['quantity'] > 0 else "BUY",
                        abs(int(position['quantity'])),
                        signal.price
                    )
                else:
                    return self.broker.close_position(self.symbol)
            return None

        # Calculate position size with 0.5% max risk
        entry_price = signal.price
        stop_loss = self.calculate_stop_loss(entry_price, signal.signal.value)
        take_profit = self.calculate_take_profit(entry_price, signal.signal.value)

        position_size = self.broker.calculate_position_size(entry_price, stop_loss)

        if position_size <= 0:
            logger.warning("Calculated position size is 0 - insufficient capital")
            return None

        logger.info(f"\n{'='*50}")
        logger.info(f"EXECUTING {signal.signal.value} SIGNAL")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Entry Price: ${entry_price:.2f}")
        logger.info(f"  Stop Loss: ${stop_loss:.2f} (-{self.max_risk_pct}%)")
        logger.info(f"  Take Profit: ${take_profit:.2f} (+{self.max_risk_pct * 2}%)")
        logger.info(f"  Position Size: {position_size} shares")
        logger.info(f"  Position Value: ${position_size * entry_price:,.2f}")
        logger.info(f"  Max Risk: ${abs(entry_price - stop_loss) * position_size:.2f}")
        logger.info(f"  Signal Strength: {signal.strength.name}")
        logger.info(f"  Reasons: {', '.join(signal.reasons)}")
        logger.info(f"{'='*50}\n")

        # Execute the trade
        action = signal.signal.value
        if isinstance(self.broker, SimulatedBroker):
            result = self.broker.place_market_order(
                self.symbol, action, position_size, entry_price
            )
        else:
            # Use bracket order with IB for automatic stop loss/take profit
            results = self.broker.place_bracket_order(
                symbol=self.symbol,
                action=action,
                quantity=position_size,
                entry_price=None,  # Market order
                stop_loss_price=stop_loss,
                take_profit_price=take_profit
            )
            result = results[0] if results else None

        if result and result.success:
            self.trade_log.append({
                'timestamp': datetime.now(),
                'signal': signal.signal.value,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantity': position_size,
                'reasons': signal.reasons
            })

        return result

    def run_once(self) -> Optional[TradeSignal]:
        """Run a single analysis and trade cycle."""
        # Fetch current data
        data = self.get_intraday_data()
        if data.empty:
            logger.warning("No data available")
            return None

        # Generate signal
        signal = self.signal_generator.generate_signal(data)
        self.last_signal = signal

        # Log current status
        logger.info(f"\n[{self.symbol}] Price: ${signal.price:.2f}")
        logger.info(f"  RSI: {signal.indicators.rsi:.1f}")
        logger.info(f"  MACD: {signal.indicators.macd:.3f} (Signal: {signal.indicators.macd_signal:.3f})")
        logger.info(f"  VWAP: ${signal.indicators.vwap:.2f}")
        logger.info(f"  Signal: {signal.signal.value} ({signal.strength.name})")

        if signal.signal != Signal.HOLD:
            logger.info(f"  Reasons: {', '.join(signal.reasons)}")

        # Execute trade if signal
        if signal.signal != Signal.HOLD:
            self.execute_trade(signal)

        return signal

    def run(self):
        """Run the trading bot continuously."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Trading Bot for {self.symbol}")
        logger.info(f"Max Risk Per Trade: {self.max_risk_pct}%")
        logger.info(f"Check Interval: {self.check_interval} seconds")
        logger.info(f"Mode: {'Paper Trading' if self.paper_trading else 'LIVE TRADING'}")
        logger.info(f"{'='*60}\n")

        # Connect to broker if IB
        if isinstance(self.broker, IBBroker):
            if not self.broker.connect():
                logger.error("Failed to connect to IB. Exiting.")
                return

        self.running = True

        try:
            while self.running:
                # Check trading hours
                if not self.should_trade():
                    logger.info("Outside trading hours. Waiting...")
                    time.sleep(60)
                    continue

                # Run analysis
                self.run_once()

                # Show account status
                summary = self.broker.get_account_summary()
                position = self.broker.get_position(self.symbol)

                logger.info(f"\n  Account Value: ${summary.get('NetLiquidation', 0):,.2f}")
                logger.info(f"  Buying Power: ${summary.get('BuyingPower', 0):,.2f}")

                if position:
                    logger.info(f"  Position: {position['quantity']} shares @ ${position['avg_cost']:.2f}")

                if isinstance(self.broker, SimulatedBroker):
                    pnl = self.broker.get_pnl()
                    logger.info(f"  Total PnL: ${pnl['total_pnl']:,.2f}")

                # Wait for next check
                logger.info(f"\n  Waiting {self.check_interval}s for next check...\n")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.running = False

        # Close position if any
        position = self.broker.get_position(self.symbol)
        if position and position['quantity'] != 0:
            logger.info(f"Closing open position: {position['quantity']} shares")
            if isinstance(self.broker, SimulatedBroker):
                # Get current price for simulation
                data = self.get_intraday_data()
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    self.broker.place_market_order(
                        self.symbol,
                        "SELL" if position['quantity'] > 0 else "BUY",
                        abs(int(position['quantity'])),
                        current_price
                    )
            else:
                self.broker.close_position(self.symbol)

        # Disconnect from IB
        if isinstance(self.broker, IBBroker):
            self.broker.disconnect()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print trading session summary."""
        logger.info(f"\n{'='*60}")
        logger.info("TRADING SESSION SUMMARY")
        logger.info(f"{'='*60}")

        summary = self.broker.get_account_summary()
        logger.info(f"Final Account Value: ${summary.get('NetLiquidation', 0):,.2f}")

        if isinstance(self.broker, SimulatedBroker):
            pnl = self.broker.get_pnl()
            logger.info(f"Realized PnL: ${pnl['realized_pnl']:,.2f}")
            logger.info(f"Unrealized PnL: ${pnl['unrealized_pnl']:,.2f}")
            logger.info(f"Total PnL: ${pnl['total_pnl']:,.2f}")
            logger.info(f"Total Trades: {pnl['total_trades']}")

            if self.broker.initial_capital > 0:
                return_pct = (pnl['total_pnl'] / self.broker.initial_capital) * 100
                logger.info(f"Return: {return_pct:.2f}%")

        logger.info(f"\nTrade Log ({len(self.trade_log)} entries):")
        for i, trade in enumerate(self.trade_log[-10:], 1):  # Last 10 trades
            logger.info(f"  {i}. {trade['timestamp'].strftime('%H:%M:%S')} - "
                       f"{trade['signal']} {trade['quantity']} @ ${trade['price']:.2f}")

        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='NVDA Intraday Trading Bot')
    parser.add_argument('--symbol', default='NVDA', help='Stock symbol to trade')
    parser.add_argument('--risk', type=float, default=0.5, help='Max risk per trade (%)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for simulation')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--paper', action='store_true', help='Simulated paper trading (default)')
    mode_group.add_argument('--ib-paper', action='store_true', help='Paper trading with IB Gateway')
    mode_group.add_argument('--live', action='store_true', help='Live trading with IB Gateway')

    parser.add_argument('--ib-port', type=int, default=4002, help='IB Gateway port')
    parser.add_argument('--once', action='store_true', help='Run once and exit')

    args = parser.parse_args()

    # Determine mode
    use_ib = args.ib_paper or args.live
    paper_trading = not args.live
    ib_port = args.ib_port

    if args.live:
        ib_port = 4001  # Live trading port
        logger.warning("\n" + "!"*60)
        logger.warning("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("!"*60 + "\n")
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != 'CONFIRM':
            logger.info("Live trading cancelled.")
            return

    bot = TradingBot(
        symbol=args.symbol,
        max_risk_pct=args.risk,
        check_interval_seconds=args.interval,
        use_ib=use_ib,
        ib_port=ib_port,
        paper_trading=paper_trading,
        initial_capital=args.capital
    )

    if args.once:
        signal = bot.run_once()
        if signal:
            print(f"\nSignal: {signal}")
        bot.print_summary()
    else:
        bot.run()


if __name__ == "__main__":
    main()
