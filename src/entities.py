from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import date, datetime
from enum import Enum

class TextBlock(BaseModel):
    section: str
    speaker: str
    text: str

class Transcript(BaseModel):
    url: str
    headline: Optional[str] = None
    event_time: datetime = None
    company_name: Optional[str] = None 
    company_ticker: Optional[str] = None
    fiscal_quarter: Optional[str] = None
    daily_volatility: Optional[float] = None
    closing_price_day_before: Optional[Tuple[date, float]] = None
    closing_price_day_of: Tuple[date, Optional[float]] = None
    closing_price_day_after: Optional[Tuple[date, float]] = None
    text_blocks: List[TextBlock] = []

class Prediction(Enum):
    Up = "UP"
    Down = "DOWN"
    Same = "SAME"
