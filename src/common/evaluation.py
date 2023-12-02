from common.entities import Transcript, StockDirection
from typing import Optional

def true_stock_direction(transcript: Transcript) -> Optional[StockDirection]:
    before_price = transcript.closing_price_day_before[1] if transcript.closing_price_day_before else None
    after_price = transcript.closing_price_day_after[1] if transcript.closing_price_day_after else None

    if before_price == None or after_price == None:
        return None 
    
    return_between_two_days = (after_price - before_price) / after_price
    
    if return_between_two_days > 0:
        return StockDirection.Up
    return StockDirection.Down