from collections import Counter
from common.entities import StockDirection, Datasets, AnswerDataPoint
from common.data_loading import load_data_splits
from tqdm import tqdm
from common.utils import llama2_token_length
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from random import seed, shuffle
import os 
from openai import OpenAI

load_dotenv()

SEED = os.environ['SEED']
seed(SEED)

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

datasets: Datasets = load_data_splits()

class OpenAIModel:

    def __init__(self, model: str) -> None:
        self._model = model
        
    def classify(self, text: str) -> StockDirection: 
        response = self._openai_request(text=text)

        if StockDirection.Up.value in response:
            return StockDirection.Up
        elif StockDirection.Down.value in response:
            return StockDirection.Down
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")

    def _openai_request(self, text: str) -> str:

        chat_completion = openai_client.chat.completions.create(
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a binary classifier with expert financial analyst knowledge, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Explain your reasoning, a line break, then either UP if you predict the stock will go up, or DOWN if you predict it will go down. You must absolutely make a prediction – don't answer with N/A.",
                },
                {
                    "role": "user",
                    "content": f"The answer from the earnings transcript is: {text}",
                }
            ],
            model=self._model,
        )
        response_lines = chat_completion.choices[0].message.content.strip().split('\n')
        return response_lines[-1].strip()


def filter_answer(answer: str, token_length_low_threshold: int = 20, token_length_high_threshold: int = 1000) -> bool:
    text_token_length = llama2_token_length(answer)
    return text_token_length >= token_length_low_threshold and text_token_length <= token_length_high_threshold

def evaluate(label: str, llm_model, datapoints: list[AnswerDataPoint]):
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

    print("Prediction Counts: ", Counter(predictions))
            
    print("="*10)
    print("Results for ", label)
    print("="*10)
    print("N of ", len(datapoints))
    print("Accuracy Score: ", accuracy_score(y_true=true_labels, y_pred=predictions))
    print("F1 Score: ", f1_score(y_true=true_labels, y_pred=predictions, pos_label='UP'))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=true_labels, y_pred=predictions, labels=["UP", "DOWN"]))


NUM_EXAMPLES_TO_EVALUATE = 1000

shuffle(datasets.test)
answer_datapoints = datasets.test[0:NUM_EXAMPLES_TO_EVALUATE]


evaluate(
    label="Test 2", 
    llm_model = OpenAIModel(model='gpt-3.5-turbo'), 
    datapoints=answer_datapoints
)

evaluate(
    label="Test 2", 
    llm_model = OpenAIModel(model='gpt-4'), 
    datapoints=answer_datapoints
)