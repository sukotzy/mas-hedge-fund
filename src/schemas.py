from enum import Enum
from pydantic import BaseModel, Field

class MarketSignal(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class Bet(BaseModel):
    ticker: str
    direction: MarketSignal
    amount: float = Field(description="Amount of capital bet on this signal")
    conviction: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Reasoning for the bet")

class AgentCapital(BaseModel):
    agent_name: str
    total_capital: float
    allocated_capital: float = 0.0
    private_capital: float = 0.0
    roi_history: list[float] = Field(default_factory=list)
