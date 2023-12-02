
from common.entities import Transcript, StockDirection, AnswerDataPoint, Datasets
from common.evaluation import true_stock_direction
import os 
from tqdm import tqdm
import pytz
from dotenv import load_dotenv
from typing import Optional, Tuple
load_dotenv()

utc = pytz.UTC

TRAINING_SET_SPLIT_RATIO = 0.6
DEVELOPMENT_SET_SPLIT_RATIO = 0.2
TEST_SET_SPLIT_RATIO = 1.0 - TRAINING_SET_SPLIT_RATIO - DEVELOPMENT_SET_SPLIT_RATIO

TRANSCRIPTS_DATA_PATH = 'transcript_data'

def _load_all_transcripts() -> list[Transcript]:
    all_transcripts: list[Transcript] = []
    for file in tqdm(os.listdir(TRANSCRIPTS_DATA_PATH), desc="Loading transcripts into memory"):
        with open(f"{TRANSCRIPTS_DATA_PATH}/{file}") as f:
            if not file.endswith('.json'):
                continue 
            all_transcripts.append(Transcript.model_validate_json(f.read()))
    return all_transcripts


def load_data_splits() -> Datasets:
    transcripts: list[Transcript] = _load_all_transcripts()

    transcripts.sort(key=lambda t: t.event_time.replace(tzinfo=utc))

    training_split_n = int(TRAINING_SET_SPLIT_RATIO * len(transcripts))
    development_split_n = training_split_n + int(DEVELOPMENT_SET_SPLIT_RATIO * len(transcripts))

    training_set_transcripts = transcripts[:training_split_n]
    development_set_transcripts = transcripts[training_split_n:development_split_n]
    test_set_transcripts = transcripts[development_split_n:]

    def datapoints_from_split(split_transcripts: list[Transcript]) -> list[AnswerDataPoint]:

        labels: list[Tuple[Transcript, Optional[StockDirection]]] = [
            (t, true_stock_direction(transcript=t))
            for t in split_transcripts
        ]

        data_points: list[AnswerDataPoint] = [
            AnswerDataPoint(answer=answer, true_label=label) 
            for transcript, label in labels if label 
            for answer in transcript.answer_texts()
        ]

        return data_points

    return Datasets(
        training=datapoints_from_split(split_transcripts=training_set_transcripts),
        development=datapoints_from_split(split_transcripts=development_set_transcripts),
        test=datapoints_from_split(split_transcripts=test_set_transcripts),
    )
