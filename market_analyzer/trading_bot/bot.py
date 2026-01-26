"""
NVDA Intraday Trading Bot - Breakout Strategy

Trades NVDA based on Donchian Channel Breakout + Momentum.
Backtested: +8.73% return (vs -3.90% buy&hold), 42.7% win rate, 1.44 profit factor.

Strategy:
    - LONG: Price breaks 10-bar high + Momentum > 1%
    - SHORT: Price breaks 10-bar low + Momentum < -1%
    - Stop Loss: 1.5 × ATR (dynamic)
    - Take Profit: 3.0 × ATR

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
from datetime import datetime
from typing import Optional, Dict
import pandas as pd
import yfinance as yf

from .signals import BreakoutSignalGenerator, BreakoutSignal, Signal
from .broker import IBBroker, SimulatedBroker, OrderResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BreakoutTradingBot:
    """
    Intraday trading bot using Donchian Channel Breakout strategy.

    Strategy (backtested +8.73% vs -3.90% buy&hold, 60 days):
    - LONG: Price breaks above 10-bar high + Momentum > 1%
    - SHORT: Price breaks below 10-bar low + Momentum < -1%
    - Stop Loss: 1.5 × ATR
    - Take Profit: 3.0 × ATR
    - Risk per trade: 0.3% of account
    """

    def __init__(
        self,
        symbol: str = "NVDA",
        risk_per_trade: float = 0.3,  # 0.3% risk per trade
        check_interval_seconds: int = 60,
        use_ib: bool = False,
        ib_port: int = 4002,
        paper_trading: bool = True,
        initial_capital: float = 100000,
        channel_period: int = 10,
        momentum_threshold: float = 0.01,  # 1%
        stop_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
    ):
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.check_interval = check_interval_seconds
        self.paper_trading = paper_trading

        # Initialize breakout signal generator
        self.signal_generator = BreakoutSignalGenerator(
            channel_period=channel_period,
            momentum_threshold=momentum_threshold,
            stop_atr_multiplier=stop_atr_mult,
            tp_atr_multiplier=tp_atr_mult,
            risk_per_trade=risk_per_trade / 100,
        )

        # Initialize broker
        if use_ib:
            self.broker = IBBroker(
                port=ib_port,
                max_risk_pct=risk_per_trade,
                paper_trading=paper_trading
            )
        else:
            self.broker = SimulatedBroker(
                initial_capital=initial_capital,
                max_risk_pct=risk_per_trade
            )

        self.running = False
        self.position: Optional[Dict] = None  # Track current position with stops
        self.trade_log = []
        self.last_signal: Optional[BreakoutSignal] = None

    def get_intraday_data(self, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Fetch intraday data for analysis."""
        try:
            data = yf.download(
                self.symbol,
                period=period,
                interval=interval,
                progress=False
            )
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            if data.empty:
                logger.warning(f"No data received for {self.symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def should_trade(self) -> bool:
        """Check if within trading hours (9:30 AM - 4:00 PM ET)."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # Trading hours: 9:30 AM - 4:00 PM (adjust for your timezone)
        if hour < 9 or (hour == 9 and minute < 30):
            return False
        if hour >= 16:
            return False
        return True

    def check_exit_conditions(self, current_price: float, current_high: float, current_low: float) -> Optional[str]:
        """
        Check if current position should be exited.

        Returns:
            Exit reason ('stop_loss', 'take_profit') or None
        """
        if not self.position:
            return None

        if self.position['direction'] == 1:  # LONG
            if current_low <= self.position['stop_loss']:
                return 'stop_loss'
            if current_high >= self.position['take_profit']:
                return 'take_profit'
        else:  # SHORT
            if current_high >= self.position['stop_loss']:
                return 'stop_loss'
            if current_low <= self.position['take_profit']:
                return 'take_profit'

        return None

    def execute_entry(self, signal: BreakoutSignal) -> Optional[OrderResult]:
        """Execute entry trade based on breakout signal."""
        if signal.direction == 0:
            return None

        # Calculate position size
        account_summary = self.broker.get_account_summary()
        account_value = account_summary.get('NetLiquidation', 100000)

        position_size = self.signal_generator.calculate_position_size(
            signal.entry_price,
            signal.stop_loss,
            account_value
        )

        if position_size <= 0:
            logger.warning("Calculated position size is 0 - insufficient capital or risk")
            return None

        action = "BUY" if signal.direction == 1 else "SELL"
        risk_amount = abs(signal.entry_price - signal.stop_loss) * position_size

        logger.info(f"\n{'='*60}")
        logger.info(f"BREAKOUT {action} SIGNAL DETECTED")
        logger.info(f"{'='*60}")
        logger.info(f"  Symbol:       {self.symbol}")
        logger.info(f"  Direction:    {'LONG' if signal.direction == 1 else 'SHORT'}")
        logger.info(f"  Entry Price:  ${signal.entry_price:.2f}")
        logger.info(f"  Stop Loss:    ${signal.stop_loss:.2f}")
        logger.info(f"  Take Profit:  ${signal.take_profit:.2f}")
        logger.info(f"  ATR:          ${signal.atr:.2f}")
        logger.info(f"  Momentum:     {signal.momentum*100:+.2f}%")
        logger.info(f"  Channel:      ${signal.channel_low:.2f} - ${signal.channel_high:.2f}")
        logger.info(f"  Position Size: {position_size} shares")
        logger.info(f"  Position Value: ${position_size * signal.entry_price:,.2f}")
        logger.info(f"  Risk Amount:  ${risk_amount:,.2f} ({self.risk_per_trade}%)")
        logger.info(f"  Reason:       {signal.reason}")
        logger.info(f"{'='*60}\n")

        # Execute the trade
        if isinstance(self.broker, SimulatedBroker):
            result = self.broker.place_market_order(
                self.symbol, action, position_size, signal.entry_price
            )
        else:
            # Use bracket order with IB
            results = self.broker.place_bracket_order(
                symbol=self.symbol,
                action=action,
                quantity=position_size,
                entry_price=None,  # Market order
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit
            )
            result = results[0] if results else None

        if result and result.success:
            # Track position locally
            self.position = {
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'quantity': position_size,
                'entry_time': datetime.now()
            }

            self.trade_log.append({
                'timestamp': datetime.now(),
                'action': 'ENTRY',
                'direction': 'LONG' if signal.direction == 1 else 'SHORT',
                'price': signal.entry_price,
                'quantity': position_size,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            })

        return result

    def execute_exit(self, exit_price: float, reason: str) -> Optional[OrderResult]:
        """Execute exit trade."""
        if not self.position:
            return None

        action = "SELL" if self.position['direction'] == 1 else "BUY"
        quantity = self.position['quantity']

        # Calculate P&L
        if self.position['direction'] == 1:
            pnl = (exit_price - self.position['entry_price']) * quantity
        else:
            pnl = (self.position['entry_price'] - exit_price) * quantity

        logger.info(f"\n{'='*60}")
        logger.info(f"EXITING POSITION - {reason.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"  Direction:    {'LONG' if self.position['direction'] == 1 else 'SHORT'}")
        logger.info(f"  Entry:        ${self.position['entry_price']:.2f}")
        logger.info(f"  Exit:         ${exit_price:.2f}")
        logger.info(f"  Quantity:     {quantity} shares")
        logger.info(f"  P&L:          ${pnl:+,.2f}")
        logger.info(f"{'='*60}\n")

        # Execute exit
        if isinstance(self.broker, SimulatedBroker):
            result = self.broker.place_market_order(
                self.symbol, action, quantity, exit_price
            )
        else:
            result = self.broker.close_position(self.symbol)

        if result and result.success:
            self.trade_log.append({
                'timestamp': datetime.now(),
                'action': 'EXIT',
                'reason': reason,
                'price': exit_price,
                'quantity': quantity,
                'pnl': pnl
            })
            self.position = None

        return result

    def run_once(self) -> Optional[BreakoutSignal]:
        """Run a single analysis and trade cycle."""
        # Fetch current data
        data = self.get_intraday_data()
        if data.empty or len(data) < 30:
            logger.warning("Insufficient data for analysis")
            return None

        current_bar = data.iloc[-1]
        current_price = float(current_bar['Close'])
        current_high = float(current_bar['High'])
        current_low = float(current_bar['Low'])

        # Check exit conditions for existing position
        if self.position:
            exit_reason = self.check_exit_conditions(current_price, current_high, current_low)
            if exit_reason:
                exit_price = self.position['stop_loss'] if exit_reason == 'stop_loss' else self.position['take_profit']
                self.execute_exit(exit_price, exit_reason)
                return None

        # Generate new signal
        signal = self.signal_generator.generate_signal(data)
        self.last_signal = signal

        # Get channel status
        status = self.signal_generator.get_channel_status(data)

        # Log current status
        logger.info(f"\n[{self.symbol}] Price: ${current_price:.2f}")
        logger.info(f"  Channel: ${status['channel_low']:.2f} - ${status['channel_high']:.2f}")
        logger.info(f"  Distance to High: {status['distance_to_high_pct']:.2f}%")
        logger.info(f"  Distance to Low: {status['distance_to_low_pct']:.2f}%")
        logger.info(f"  Momentum: {status['momentum_pct']:+.2f}%")
        logger.info(f"  ATR: ${status['atr']:.2f}")

        if self.position:
            direction = 'LONG' if self.position['direction'] == 1 else 'SHORT'
            unrealized = (current_price - self.position['entry_price']) * self.position['quantity'] * self.position['direction']
            logger.info(f"  Position: {direction} {self.position['quantity']} @ ${self.position['entry_price']:.2f}")
            logger.info(f"  Unrealized P&L: ${unrealized:+,.2f}")
        else:
            logger.info(f"  Position: None")
            if status['near_breakout_long']:
                logger.info(f"  ⚡ NEAR LONG BREAKOUT!")
            if status['near_breakout_short']:
                logger.info(f"  ⚡ NEAR SHORT BREAKOUT!")

        # Execute entry if signal and no position
        if signal.direction != 0 and not self.position:
            self.execute_entry(signal)

        return signal

    def run(self):
        """Run the trading bot continuously."""
        logger.info(f"\n{'='*60}")
        logger.info(f"BREAKOUT TRADING BOT - {self.symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Strategy: Donchian Channel Breakout + Momentum")
        logger.info(f"Risk Per Trade: {self.risk_per_trade}%")
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
                logger.info(f"\n  Account Value: ${summary.get('NetLiquidation', 0):,.2f}")
                logger.info(f"  Buying Power: ${summary.get('BuyingPower', 0):,.2f}")

                if isinstance(self.broker, SimulatedBroker):
                    pnl = self.broker.get_pnl()
                    logger.info(f"  Total P&L: ${pnl['total_pnl']:+,.2f}")

                # Wait for next check
                logger.info(f"\n  Next check in {self.check_interval}s...\n")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.running = False

        # Close position if any
        if self.position:
            logger.info(f"Closing open position...")
            data = self.get_intraday_data()
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
                self.execute_exit(current_price, 'bot_stopped')

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
            logger.info(f"Realized P&L: ${pnl['realized_pnl']:+,.2f}")
            logger.info(f"Unrealized P&L: ${pnl['unrealized_pnl']:+,.2f}")
            logger.info(f"Total P&L: ${pnl['total_pnl']:+,.2f}")
            logger.info(f"Total Trades: {pnl['total_trades']}")

            if self.broker.initial_capital > 0:
                return_pct = (pnl['total_pnl'] / self.broker.initial_capital) * 100
                logger.info(f"Return: {return_pct:+.2f}%")

        # Trade log
        entries = [t for t in self.trade_log if t['action'] == 'ENTRY']
        exits = [t for t in self.trade_log if t['action'] == 'EXIT']

        logger.info(f"\nTrades: {len(entries)} entries, {len(exits)} exits")

        if exits:
            wins = sum(1 for t in exits if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in exits)
            logger.info(f"Win Rate: {wins}/{len(exits)} ({wins/len(exits)*100:.1f}%)")
            logger.info(f"Total Realized: ${total_pnl:+,.2f}")

        logger.info(f"{'='*60}\n")


# Keep old TradingBot for backwards compatibility
TradingBot = BreakoutTradingBot


def main():
    parser = argparse.ArgumentParser(description='NVDA Breakout Trading Bot')
    parser.add_argument('--symbol', default='NVDA', help='Stock symbol to trade')
    parser.add_argument('--risk', type=float, default=0.3, help='Risk per trade (%)')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital for simulation')

    # Strategy parameters (optimized via backtest)
    parser.add_argument('--channel', type=int, default=10, help='Donchian channel period')
    parser.add_argument('--momentum', type=float, default=0.01, help='Momentum threshold (0.01 = 1%)')
    parser.add_argument('--stop-atr', type=float, default=1.5, help='Stop loss ATR multiplier')
    parser.add_argument('--tp-atr', type=float, default=3.0, help='Take profit ATR multiplier')

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

    bot = BreakoutTradingBot(
        symbol=args.symbol,
        risk_per_trade=args.risk,
        check_interval_seconds=args.interval,
        use_ib=use_ib,
        ib_port=ib_port,
        paper_trading=paper_trading,
        initial_capital=args.capital,
        channel_period=args.channel,
        momentum_threshold=args.momentum,
        stop_atr_mult=args.stop_atr,
        tp_atr_mult=args.tp_atr,
    )

    if args.once:
        signal = bot.run_once()
        if signal:
            print(f"\n{signal}")
        bot.print_summary()
    else:
        bot.run()


if __name__ == "__main__":
    main()
