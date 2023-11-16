from entities import Transcript, Prediction
from typing import Optional

def true_stock_direction(transcript: Transcript, standard_deviation_multiples: float) -> Optional[Prediction]:
    before_price = transcript.closing_price_day_before[1] if transcript.closing_price_day_before else None
    after_price = transcript.closing_price_day_after[1] if transcript.closing_price_day_after else None

    if before_price == None or after_price == None:
        return None 
    
    low_threshold = before_price - standard_deviation_multiples * transcript.daily_volatility
    high_threshold = before_price + standard_deviation_multiples * transcript.daily_volatility

    if after_price > high_threshold:
        return Prediction.Up
    elif after_price < low_threshold:
        return Prediction.Down
    
    return Prediction.Same
