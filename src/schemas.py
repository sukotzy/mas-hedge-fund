from enum import Enum
from pydantic import BaseModel, Field

class MarketSignal(Enum):
    LONG = "long"
    SHORT = "short"

class Bet(BaseModel):
    ticker: str
    direction: MarketSignal
    amount: float = Field(description="Amount of capital bet on this signal")
    conviction: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Reasoning for the bet")

class AgentCapital(BaseModel):
    agent_name: str
    allocated_capital: float = 0.0  # Kept for backwards compatibility, represents 'external'
    internal_capital: float = 0.0   # Renamed from private_capital
    external_capital: float = 0.0   # Explicit external capital
    roi_history: list[float] = Field(default_factory=list)

    @property
    def total_capital(self) -> float:
        return self.internal_capital + self.external_capital

from typing import List, Literal
class Allocation(BaseModel):
    ticker: str
    direction: Literal["long", "short"]
    amount: float = Field(description="Capital allocated (0-100)")
    reasoning: str

class PortfolioDecision(BaseModel):
    allocations: List[Allocation]
    # Implicit Constraint: sum(a.amount for a in allocations) == 100.0
