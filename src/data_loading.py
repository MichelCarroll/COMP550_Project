
from entities import Transcript, Prediction, DataPoint, Datasets
from evaluation import true_stock_direction
import os 
from tqdm import tqdm
import pytz
from dotenv import load_dotenv
from typing import Optional
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


def load_data_splits(number_std_devs_threshold: int = 2) -> Datasets:
    transcripts: list[Transcript] = _load_all_transcripts()
    labels: list[Transcript, Optional[Prediction]] = [
        (t, true_stock_direction(transcript=t, standard_deviation_multiples=number_std_devs_threshold))
        for t in transcripts
    ]

    data_points: list[DataPoint] = [DataPoint(transcript=t, true_label=label) for t, label in labels if label]

    transcripts.sort(key=lambda t: t.event_time.replace(tzinfo=utc))

    training_split_n = int(TRAINING_SET_SPLIT_RATIO * len(transcripts))
    development_split_n = training_split_n + int(DEVELOPMENT_SET_SPLIT_RATIO * len(transcripts))
    
    training_set_transcripts = data_points[:training_split_n]
    development_set_transcripts = data_points[training_split_n:development_split_n]
    test_set_transcripts = data_points[development_split_n:]

    return Datasets(
        training=training_set_transcripts,
        development=development_set_transcripts,
        test=test_set_transcripts,
    )
