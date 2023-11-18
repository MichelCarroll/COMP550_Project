from datasets import Dataset, DatasetDict
from entities import Prediction, DataPoint
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


def huggingface_dataset(datapoints: list[DataPoint], number_of_examples: int):

    all_example_sources = [
        (answer, datapoint.true_label) 
        for datapoint in datapoints
        for answer in datapoint.transcript.answer_texts()
    ]
    random.shuffle(all_example_sources)

    completions: list[str] = []
    
    for answer, true_label in tqdm(all_example_sources, desc=f"Preparing examples", total=number_of_examples):
        completion = f"""[INST] <<SYS>>You are a financial analyst, determining what effect an earnings call will have on the company's stock price, based on a summary of one. Be as critical and skeptical as possible, and remember that the market may disproportionally react to small unexpected details. Respond with one of: {Prediction.Up.value}, {Prediction.Down.value}, {Prediction.Same.value}<</SYS>> {answer}[/INST]
    Prediction (one of UP, DOWN, SAME): {true_label.value}
    """
        token_length = llama2_token_length(completion)
        if token_length > MAX_EXAMPLE_TOKEN_LENGTH:
            continue 

        completions.append(completion)

        if len(completions) >= number_of_examples:
            break

    return Dataset.from_dict({'completion': completions})

train_split = huggingface_dataset(datapoints=dataset.training, number_of_examples=NUMBER_TRAINING_EXAMPLES)
development_split = huggingface_dataset(datapoints=dataset.development, number_of_examples=len(dataset.development))
test_split = huggingface_dataset(datapoints=dataset.test, number_of_examples=len(dataset.test))

dataset_dict = DatasetDict({
    "train": train_split,
    "development": development_split,
    "test": test_split,
})

dataset_dict.push_to_hub("michelcarroll/llama2-earnings-stock-prediction-fine-tune")