from __future__ import annotations

from typing import Dict, Mapping, Mapping as _MappingAny

from .types import PortfolioSnapshot, PositionState, TickerRealizedGains
import math


def calculate_portfolio_value(portfolio: Portfolio, current_prices: Mapping[str, float]) -> float:
    """Compute total portfolio value identical to Backtester.calculate_portfolio_value.

    total_value = cash + market value of longs - market value of shorts
    """
    components = [portfolio.get_cash(), portfolio.get_margin_used()]
    positions = portfolio.get_positions()
    for ticker, pos in positions.items():
        price = float(current_prices.get(ticker, 0.0))
        if pos["long"] > 0:
            components.append(pos["long"] * price)
        if pos["short"] > 0:
            components.append(-pos["short"] * price)
            
    return round(math.fsum(components), 4)


def compute_exposures(portfolio: Portfolio, current_prices: Mapping[str, float]) -> Dict[str, float]:
    """Compute long/short/gross/net exposures and long/short ratio.

    Mirrors the calculations performed in src/backtester.py run loop.
    """
    positions = portfolio.get_positions()
    long_comps = []
    short_comps = []
    
    for ticker, pos in positions.items():
        price = float(current_prices.get(ticker, 0.0))
        if pos["long"] > 0:
            long_comps.append(pos["long"] * price)
        if pos["short"] > 0:
            short_comps.append(pos["short"] * price)

    long_exposure = round(math.fsum(long_comps), 4)
    short_exposure = round(math.fsum(short_comps), 4)

    gross_exposure = round(long_exposure + short_exposure, 4)
    net_exposure = round(long_exposure - short_exposure, 4)
    if short_exposure > 1e-9:
        long_short_ratio = round(long_exposure / short_exposure, 4)
    else:
        long_short_ratio = float("inf")

    return {
        "Long Exposure": long_exposure,
        "Short Exposure": short_exposure,
        "Gross Exposure": gross_exposure,
        "Net Exposure": net_exposure,
        "Long/Short Ratio": long_short_ratio,
    }



def compute_portfolio_summary(
    *,
    portfolio: Portfolio,
    total_value: float,
    initial_value: float | None,
    performance_metrics: _MappingAny[str, float | None],
) -> Dict[str, float | None]:
    """Compute portfolio summary fields in a pure, testable function.

    Returns a dict with keys matching the arguments used by format_backtest_row
    for the summary row (excluding is_summary and date-specific fields).
    """
    cash_balance = portfolio.get_cash()
    total_position_value = total_value - cash_balance
    if initial_value and initial_value != 0:
        return_pct = (total_value / initial_value - 1.0) * 100.0
    else:
        return_pct = 0.0

    return {
        "total_value": float(total_value),
        "return_pct": float(return_pct),
        "cash_balance": float(cash_balance),
        "total_position_value": float(total_position_value),
        "sharpe_ratio": performance_metrics.get("sharpe_ratio"),
        "sortino_ratio": performance_metrics.get("sortino_ratio"),
        "max_drawdown": performance_metrics.get("max_drawdown"),
    }

