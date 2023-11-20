from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import date, datetime
from enum import Enum
from transformers import LlamaTokenizer
 
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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
    
    def answer_texts(self) -> list[str]:
        answer_texts: list[str] = []
        last_block_was_analyst: bool = False

        for block in self.text_blocks:
            if block.section == 'Questions and Answers':
                if 'Operator' in block.speaker:
                    continue 
                elif last_block_was_analyst:
                    answer_texts.append(block.text)
                    last_block_was_analyst = False 
                elif 'Analyst' in block.speaker:
                    last_block_was_analyst = True

        return answer_texts

class StockDirection(Enum):
    Up = "UP"
    Down = "DOWN"

class AnswerDataPoint(BaseModel):
    answer: str
    true_label: StockDirection


class Datasets(BaseModel):
    training: list[AnswerDataPoint]
    development: list[AnswerDataPoint]
    test: list[AnswerDataPoint]