from datasets import Dataset, DatasetDict
from common.entities import StockDirection, AnswerDataPoint
from collections import Counter
from common.data_loading import load_data_splits
from dotenv import load_dotenv
from common.utils import llama2_token_length
from tqdm import tqdm
import random 
import os 
from random import shuffle
from typing import Optional

load_dotenv()

SEED = os.environ['SEED']

random.seed(SEED)

MAX_EXAMPLE_TOKEN_LENGTH = 500

dataset = load_data_splits()


def huggingface_dataset(label: str, datapoints: list[AnswerDataPoint], balance_classes: bool = False, include_label: bool = False, max_examples: Optional[int] = None):

    print(f"Preparing split '{label}'")

    completion_and_datapoint = [
        (example, f"""[INST] <<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Be as critical and skeptical as possible. Respond with {StockDirection.Up.value} or {StockDirection.Down.value}<</SYS>> {example.answer}[/INST]
    Direction (UP or DOWN): {example.true_label.value if include_label else ""}
    """) 
        for example in datapoints
    ]

    filtered_completion_and_datapoint: list[tuple[AnswerDataPoint, str]] = []
    for (c,d) in tqdm(completion_and_datapoint, desc="Preparing examples"):
        if llama2_token_length(d) <= MAX_EXAMPLE_TOKEN_LENGTH:
            filtered_completion_and_datapoint.append((c,d))  

    if balance_classes:
        shuffle(filtered_completion_and_datapoint)
        up_label_examples = [(e,c) for e,c in filtered_completion_and_datapoint if e.true_label == StockDirection.Up]
        down_label_examples = [(e,c) for e,c in filtered_completion_and_datapoint if e.true_label == StockDirection.Down]
        num_minority_class = min(len(up_label_examples), len(down_label_examples))
        # Interleaved to make choosing a subset of balanced training examples easier downstream
        filtered_completion_and_datapoint = [val for pair in zip(up_label_examples[:num_minority_class], down_label_examples[:num_minority_class]) for val in pair]

    if max_examples:
        filtered_completion_and_datapoint = filtered_completion_and_datapoint[:max_examples]

    print("Counts", Counter([e.true_label for e,_ in filtered_completion_and_datapoint]))

    return Dataset.from_dict({'completion': [c for _,c in filtered_completion_and_datapoint], 'label': [l.true_label.value for l,_ in filtered_completion_and_datapoint]})

train_split = huggingface_dataset(label="train",datapoints=dataset.training, include_label=True, balance_classes=True)
development_split = huggingface_dataset(label="development", balance_classes=True,datapoints=dataset.development)
test_split = huggingface_dataset(label="test",datapoints=dataset.test, balance_classes=True, max_examples=1000)

dataset_dict = DatasetDict({
    "train": train_split,
    "development": development_split,
    "test": test_split,
})

dataset_dict.push_to_hub("michelcarroll/llama2-earnings-stock-prediction-fine-tune-v3")