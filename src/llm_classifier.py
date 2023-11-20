from entities import StockDirection, Datasets, AnswerDataPoint
from data_loading import load_data_splits
from collections import Counter
from tqdm import tqdm
import json
from utils import llama2_token_length
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from random import seed, shuffle
import os 
load_dotenv()

SEED = os.environ['SEED']
seed(SEED)


import requests

NUMBER_STANDARD_DEVIATIONS_THRESHOLD = 2

datasets: Datasets = load_data_splits()


class LLMModel:

    def __init__(self, model: str, max_question_summary_token_length: int = 100, classification_temperature: float = 0, summarization_temperature: float = 0.2) -> None:
        self._model = model
        self._classification_temperature = classification_temperature
        self._summarization_temperature = summarization_temperature
        self._max_question_summary_token_length = max_question_summary_token_length

    def classify(self, text: str) -> StockDirection: 

        if self._model.startswith('llama'):
            prompt = f"""[INST] <<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Be as critical and skeptical as possible. Respond with {StockDirection.Up.value} or {StockDirection.Down.value}<</SYS>> {text}[/INST]
Direction (UP or DOWN): """
        elif self._model.startswith('mistral'):
            prompt = f"""[INST]You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Be as critical and skeptical as possible. Respond with {StockDirection.Up.value} or {StockDirection.Down.value}. {text}[/INST]
Direction (UP or DOWN):"""

        output = requests.post(
            url='http://localhost:11434/api/generate',
            data=json.dumps({
                "model": self._model,
                "stream": False,
                "options": {
                    "temperature": self._classification_temperature,
                    "num_predict": 10,
                    "seed": int(SEED),
                    "stop": ["\n"],
                },
                "template": "{{ .Prompt }}",
                "prompt": prompt})
        )
        response_json = output.json()
        if 'response' not  in response_json:
            raise Exception(f"No response in Ollama output: {response_json}")
        response = response_json['response']
        if StockDirection.Up.value in response:
            return StockDirection.Up
        elif StockDirection.Down.value in response:
            return StockDirection.Down
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")


def filter_answer(answer: str, token_length_low_threshold: int = 20, token_length_high_threshold: int = 1000) -> bool:
    text_token_length = llama2_token_length(answer)
    return text_token_length >= token_length_low_threshold and text_token_length <= token_length_high_threshold

NUM_EXAMPLES_TO_EVALUATE = 1000

shuffle(datasets.development)
answer_datapoints = datasets.development[0:NUM_EXAMPLES_TO_EVALUATE]

print(Counter([d.true_label for d in answer_datapoints]))

def evaluate(label: str, llm_model: LLMModel, datapoints: list[AnswerDataPoint]):
    predictions: list[StockDirection] = []
    true_labels: list[StockDirection] = []

    for datapoint in tqdm(datapoints, desc="Evaluating"):
        try:
            result = llm_model.classify(text=datapoint.answer)
        except Exception as e:
            print("ERROR", e.args[0])
            continue 
        if result:
            predictions.append(result.value)
            true_labels.append(datapoint.true_label.value)

    print("="*10)
    print("Results for ", label)
    print("="*10)
    print("N of ", len(datapoints))
    print("Accuracy Score: ", accuracy_score(y_true=true_labels, y_pred=predictions))
    print("F1 Score: ", f1_score(y_true=true_labels, y_pred=predictions, pos_label='UP'))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=true_labels, y_pred=predictions, labels=["UP", "DOWN"]))

evaluate(label="Base Llama2 7B", llm_model = LLMModel(model='llama2:7b'), datapoints=answer_datapoints)
evaluate(label="Base Mistral (instruct) 7B", llm_model = LLMModel(model='mistral'), datapoints=answer_datapoints)
evaluate(label="Fine-Tuned Llama2 7B (1000 examples)", llm_model = LLMModel(model='llama:7b-fine-tune-v4'), datapoints=answer_datapoints)
