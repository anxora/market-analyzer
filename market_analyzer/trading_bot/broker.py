"""
Interactive Brokers API integration using ib_insync.

Requires:
    pip install ib_insync

Connection:
    - IB Gateway or TWS must be running
    - API connections must be enabled in IB Gateway/TWS settings
    - Default ports: TWS=7497 (paper), 7496 (live) | Gateway=4002 (paper), 4001 (live)
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import time

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from ib_insync import Contract

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Trade, Position
    from ib_insync import util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[int]
    fill_price: Optional[float]
    quantity: int
    status: str
    message: str
    timestamp: datetime


class IBBroker:
    """
    Interactive Brokers broker interface.

    Handles connection, order execution, and position management.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,  # 4002=Gateway Paper, 4001=Gateway Live, 7497=TWS Paper, 7496=TWS Live
        client_id: int = 1,
        max_risk_pct: float = 0.5,  # Maximum risk per trade (0.5%)
        paper_trading: bool = True
    ):
        if not IB_AVAILABLE:
            raise ImportError("ib_insync is required. Install with: pip install ib_insync")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.max_risk_pct = max_risk_pct
        self.paper_trading = paper_trading

        self.ib = IB()
        self._connected = False

    def connect(self) -> bool:
        """Connect to IB Gateway/TWS."""
        try:
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=False
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")

            # Log account info
            accounts = self.ib.managedAccounts()
            logger.info(f"Managed accounts: {accounts}")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self.ib.isConnected()

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary including buying power and cash."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        summary = {}
        account_values = self.ib.accountSummary()

        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'AvailableFunds']:
                summary[av.tag] = float(av.value)

        return summary

    def get_buying_power(self) -> float:
        """Get available buying power."""
        summary = self.get_account_summary()
        return summary.get('BuyingPower', 0.0)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == symbol:
                return {
                    'symbol': symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.position * pos.avgCost
                }
        return None

    def get_all_positions(self) -> List[Dict]:
        """Get all current positions."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        positions = []
        for pos in self.ib.positions():
            positions.append({
                'symbol': pos.contract.symbol,
                'quantity': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.position * pos.avgCost
            })
        return positions

    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        """Create a stock contract."""
        return Stock(symbol, exchange, currency)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current market price for a symbol."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)

        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)  # Wait for data

        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        return price if price > 0 else None

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "5 mins"
    ) -> pd.DataFrame:
        """
        Get historical bar data.

        Args:
            symbol: Stock symbol
            duration: Time span (e.g., "5 D", "1 M", "1 Y")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,  # Regular trading hours only
            formatDate=1
        )

        if not bars:
            return pd.DataFrame()

        df = util.df(bars)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df.set_index('date')
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_value: Optional[float] = None
    ) -> int:
        """
        Calculate position size based on max risk percentage.

        Risk per share = |entry_price - stop_loss_price|
        Max risk amount = account_value * max_risk_pct / 100
        Position size = Max risk amount / Risk per share

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_value: Account value (fetched if not provided)

        Returns:
            Number of shares to buy/sell
        """
        if account_value is None:
            summary = self.get_account_summary()
            account_value = summary.get('NetLiquidation', 100000)

        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            # Default to 0.5% of price as risk
            risk_per_share = entry_price * (self.max_risk_pct / 100)

        max_risk_amount = account_value * (self.max_risk_pct / 100)
        position_size = int(max_risk_amount / risk_per_share)

        # Ensure we don't use more than available buying power
        buying_power = self.get_buying_power()
        max_shares = int(buying_power / entry_price)

        return min(position_size, max_shares)

    def place_bracket_order(
        self,
        symbol: str,
        action: str,  # "BUY" or "SELL"
        quantity: int,
        entry_price: Optional[float] = None,  # None = market order
        stop_loss_price: float = None,
        take_profit_price: float = None
    ) -> List[OrderResult]:
        """
        Place a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Stock symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            entry_price: Limit price for entry (None = market order)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price

        Returns:
            List of OrderResult for each leg
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)

        results = []

        # Create bracket order
        bracket = self.ib.bracketOrder(
            action=action,
            quantity=quantity,
            limitPrice=entry_price if entry_price else 0,
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price
        )

        # If market order, modify the parent order
        if entry_price is None:
            bracket[0].orderType = 'MKT'
            bracket[0].lmtPrice = 0

        # Place all orders
        for order in bracket:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)

            results.append(OrderResult(
                success=trade.orderStatus.status not in ['Cancelled', 'ApiCanceled'],
                order_id=trade.order.orderId,
                fill_price=trade.orderStatus.avgFillPrice if trade.orderStatus.filled > 0 else None,
                quantity=quantity,
                status=trade.orderStatus.status,
                message=f"{order.orderType} order placed",
                timestamp=datetime.now()
            ))

        return results

    def place_market_order(
        self,
        symbol: str,
        action: str,
        quantity: int
    ) -> OrderResult:
        """Place a simple market order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)

        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)

        # Wait for fill
        timeout = 30
        start = time.time()
        while trade.orderStatus.status not in ['Filled', 'Cancelled', 'ApiCanceled']:
            self.ib.sleep(0.5)
            if time.time() - start > timeout:
                break

        return OrderResult(
            success=trade.orderStatus.status == 'Filled',
            order_id=trade.order.orderId,
            fill_price=trade.orderStatus.avgFillPrice,
            quantity=int(trade.orderStatus.filled),
            status=trade.orderStatus.status,
            message=f"Market {action} order",
            timestamp=datetime.now()
        )

    def place_stop_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_price: float
    ) -> OrderResult:
        """Place a stop order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)

        order = StopOrder(action, quantity, stop_price)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)

        return OrderResult(
            success=trade.orderStatus.status not in ['Cancelled', 'ApiCanceled'],
            order_id=trade.order.orderId,
            fill_price=None,  # Not filled yet
            quantity=quantity,
            status=trade.orderStatus.status,
            message=f"Stop {action} order at ${stop_price:.2f}",
            timestamp=datetime.now()
        )

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                return True
        return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        self.ib.reqGlobalCancel()
        return len(self.ib.openTrades())

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """Close an existing position."""
        position = self.get_position(symbol)
        if not position or position['quantity'] == 0:
            return None

        action = "SELL" if position['quantity'] > 0 else "BUY"
        quantity = abs(int(position['quantity']))

        return self.place_market_order(symbol, action, quantity)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class SimulatedBroker:
    """
    Simulated broker for paper trading without IB connection.
    Useful for testing and backtesting.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_risk_pct: float = 0.5
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_pct = max_risk_pct
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.trades: List[Dict] = []

    def is_connected(self) -> bool:
        return True

    def get_account_summary(self) -> Dict[str, Any]:
        position_value = sum(
            p['quantity'] * p['current_price']
            for p in self.positions.values()
        )
        return {
            'NetLiquidation': self.capital + position_value,
            'TotalCashValue': self.capital,
            'BuyingPower': self.capital,
            'AvailableFunds': self.capital
        }

    def get_buying_power(self) -> float:
        return self.capital

    def get_position(self, symbol: str) -> Optional[Dict]:
        return self.positions.get(symbol)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_value: Optional[float] = None
    ) -> int:
        if account_value is None:
            account_value = self.capital

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            risk_per_share = entry_price * (self.max_risk_pct / 100)

        max_risk_amount = account_value * (self.max_risk_pct / 100)
        position_size = int(max_risk_amount / risk_per_share)

        max_shares = int(self.capital / entry_price)
        return min(position_size, max_shares)

    def place_market_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        current_price: float
    ) -> OrderResult:
        """Simulate a market order."""
        if action == "BUY":
            cost = quantity * current_price
            if cost > self.capital:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    quantity=0,
                    status="Rejected",
                    message="Insufficient funds",
                    timestamp=datetime.now()
                )

            self.capital -= cost
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos['quantity'] + quantity
                total_cost = (pos['quantity'] * pos['avg_cost']) + cost
                pos['quantity'] = total_qty
                pos['avg_cost'] = total_cost / total_qty
                pos['current_price'] = current_price
            else:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_cost': current_price,
                    'current_price': current_price
                }

        else:  # SELL
            if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    quantity=0,
                    status="Rejected",
                    message="Insufficient position",
                    timestamp=datetime.now()
                )

            pos = self.positions[symbol]
            pnl = (current_price - pos['avg_cost']) * quantity
            self.capital += quantity * current_price
            pos['quantity'] -= quantity

            if pos['quantity'] == 0:
                del self.positions[symbol]

            self.trades.append({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'pnl': pnl,
                'timestamp': datetime.now()
            })

        return OrderResult(
            success=True,
            order_id=len(self.orders) + 1,
            fill_price=current_price,
            quantity=quantity,
            status="Filled",
            message=f"Simulated {action}",
            timestamp=datetime.now()
        )

    def get_pnl(self) -> Dict:
        """Get profit/loss summary."""
        realized_pnl = sum(t['pnl'] for t in self.trades)
        unrealized_pnl = sum(
            (p['current_price'] - p['avg_cost']) * p['quantity']
            for p in self.positions.values()
        )
        return {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'total_trades': len(self.trades)
        }


if __name__ == "__main__":
    print("Testing SimulatedBroker...")

    broker = SimulatedBroker(initial_capital=100000, max_risk_pct=0.5)

    # Test position sizing with 0.5% risk
    entry = 150.0
    stop = entry * 0.995  # 0.5% below entry
    size = broker.calculate_position_size(entry, stop)
    print(f"\nPosition sizing for NVDA @ ${entry}:")
    print(f"  Stop loss: ${stop:.2f} (-0.5%)")
    print(f"  Max risk: ${100000 * 0.005:.2f}")
    print(f"  Position size: {size} shares")
    print(f"  Position value: ${size * entry:,.2f}")

    # Simulate a trade
    result = broker.place_market_order("NVDA", "BUY", size, entry)
    print(f"\n{result}")

    # Check position
    pos = broker.get_position("NVDA")
    print(f"\nPosition: {pos}")

    # Simulate exit
    exit_price = 152.0
    result = broker.place_market_order("NVDA", "SELL", size, exit_price)
    print(f"\n{result}")

    # Check PnL
    pnl = broker.get_pnl()
    print(f"\nPnL Summary: {pnl}")
