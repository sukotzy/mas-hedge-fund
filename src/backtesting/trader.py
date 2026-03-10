from __future__ import annotations

from .portfolio import Portfolio
from .types import ActionLiteral, Action


class TradeExecutor:
    """Executes trades against a Portfolio with Backtester-identical semantics.
    Supports seamless Long-to-Short and Short-to-Long flips."""

    def execute_trade(
        self,
        ticker: str,
        action: ActionLiteral,
        quantity: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> int:
        if quantity is None or quantity <= 0:
            return 0

        # Coerce to enum if strings provided
        try:
            action_enum = Action(action) if not isinstance(action, Action) else action
        except Exception:
            action_enum = Action.HOLD

        qty = int(quantity)
        price = float(current_price)

        if action_enum == Action.BUY:
            return portfolio.apply_long_buy(ticker, qty, price)
            
        if action_enum == Action.SELL:
            # 1. 先尝试平掉现有的多头仓位
            executed = portfolio.apply_long_sell(ticker, qty, price)
            remainder = qty - executed
            # 2. 如果还有剩余没卖完的（说明要翻空），直接开空单！
            if remainder > 0:
                executed += portfolio.apply_short_open(ticker, remainder, price)
            return executed
            
        if action_enum == Action.SHORT:
            return portfolio.apply_short_open(ticker, qty, price)
            
        if action_enum == Action.COVER:
            # 1. 先尝试平掉现有的空头仓位
            executed = portfolio.apply_short_cover(ticker, qty, price)
            remainder = qty - executed
            # 2. 如果还有剩余没平完的（说明要翻多），直接开多单买入！
            if remainder > 0:
                executed += portfolio.apply_long_buy(ticker, remainder, price)
            return executed

        # hold or unknown action
        return 0
