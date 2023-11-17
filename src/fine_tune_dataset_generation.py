from datasets import load_dataset, Dataset
from entities import Prediction
from data_loading import load_data_splits
from dotenv import load_dotenv
load_dotenv()
from utils import llama2_token_length
from tqdm import tqdm
import random 
import os 

SEED = os.environ['SEED']

random.seed(SEED)

MAX_EXAMPLE_TOKEN_LENGTH = 500
NUMBER_TRAINING_EXAMPLES = 10000

dataset = load_data_splits()
completions: list[str] = []

all_training_example_sources = [
    (answer, datapoint.true_label) 
    for datapoint in dataset.training 
    for answer in datapoint.transcript.answer_texts()
]
random.shuffle(all_training_example_sources)


completions: list[str] = []

for answer, true_label in tqdm(all_training_example_sources, desc="Preparing training examples", total=NUMBER_TRAINING_EXAMPLES):
    completion = f"""[INST] <<SYS>>You are a financial analyst, determining what effect an earnings call will have on the company's stock price, based on a summary of one. Be as critical and skeptical as possible, and remember that the market may disproportionally react to small unexpected details. Respond with one of: {Prediction.Up.value}, {Prediction.Down.value}, {Prediction.Same.value}<</SYS>> {answer}[/INST]
Prediction (one of UP, DOWN, SAME): {true_label.value}
"""
    token_length = llama2_token_length(completion)
    if token_length > MAX_EXAMPLE_TOKEN_LENGTH:
        continue 

    completions.append(completion)

    if len(completions) >= NUMBER_TRAINING_EXAMPLES:
        break

dataset = Dataset.from_dict({'completion': completions})

dataset.push_to_hub("michelcarroll/llama2-earnings-stock-prediction-fine-tune")