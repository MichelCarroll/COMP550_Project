from collections import Counter
from common.entities import StockDirection, Datasets, AnswerDataPoint
from common.data_loading import load_data_splits
from tqdm import tqdm
from common.utils import llama2_token_length
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from random import seed, shuffle
import os 
from common.transformer_operations import run_model, load_fine_tuned_model, load_base_model, clean_from_gpu
import huggingface_hub
from datasets import load_dataset

load_dotenv()

SEED = os.environ['SEED']
seed(SEED)

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']
huggingface_hub.login(token=HUGGINGFACE_TOKEN)

dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-v3'

NUM_EXAMPLES_TO_EVALUATE = 100

split_name = 'test'

class LLMModel:

    def __init__(self, model, tokenizer, max_question_summary_token_length: int = 100, classification_temperature: float = 0.05, summarization_temperature: float = 0.2) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._classification_temperature = classification_temperature
        self._summarization_temperature = summarization_temperature
        self._max_question_summary_token_length = max_question_summary_token_length

    def classify(self, text: str) -> StockDirection: 
        response = self._transformer_request(text=text)

        if StockDirection.Up.value in response:
            return StockDirection.Up
        elif StockDirection.Down.value in response:
            return StockDirection.Down
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")

    def _transformer_request(self, text: str) -> str:
        prompt = f"""[INST] <<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Respond with {StockDirection.Up.value} or {StockDirection.Down.value}<</SYS>> {text}[/INST]
Direction (UP or DOWN): """
        response = run_model(
            model=self._model, 
            tokenizer=self._tokenizer, 
            max_new_tokens=10, 
            prompt=prompt, 
            return_full_text=False
        )
        
        return response

def filter_answer(answer: str, token_length_low_threshold: int = 20, token_length_high_threshold: int = 1000) -> bool:
    text_token_length = llama2_token_length(answer)
    return text_token_length >= token_length_low_threshold and text_token_length <= token_length_high_threshold

def evaluate(label: str, llm_model: LLMModel, datapoints: list[AnswerDataPoint]):
    predictions: list[StockDirection] = []
    true_labels: list[StockDirection] = []
    misses = 0

    for datapoint in tqdm(datapoints, desc="Evaluating"):
        try:
            result = llm_model.classify(text=datapoint.answer)
        except Exception as e:
            misses += 1
            print("ERROR", e.args[0])
            continue 
        if result:
            predictions.append(result.value)
            true_labels.append(datapoint.true_label.value)

    print("Prediction Counts: ", Counter(predictions))
            
    print("="*1)
    print("Misses", misses)
    print("Results for ", label)
    print("="*10)
    print("N of ", len(datapoints))
    print("Accuracy Score: ", accuracy_score(y_true=true_labels, y_pred=predictions))
    print("F1 Score: ", f1_score(y_true=true_labels, y_pred=predictions, pos_label='UP'))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=true_labels, y_pred=predictions, labels=["UP", "DOWN"]))


answer_datapoints = load_dataset(dataset_name, split=f"{split_name}[0:{NUM_EXAMPLES_TO_EVALUATE}]")

# from collections import Counter
# print(Counter([a.true_label.value for a in answer_datapoints]))
# > Counter({'UP': 527, 'DOWN': 473})

for r in [6,8]:
    model, tokenizer = load_fine_tuned_model(f'llama2-1000examples--r{r}-a8-e1')
    evaluate(
        label=f"R{r}", 
        llm_model = LLMModel(model=model, tokenizer=tokenizer), 
        datapoints=answer_datapoints
    )
    clean_from_gpu(model)
    clean_from_gpu(tokenizer)