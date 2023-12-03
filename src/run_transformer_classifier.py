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

dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-binary'

class LLMModel:

    def __init__(self, model, tokenizer, max_question_summary_token_length: int = 100, classification_temperature: float = 0.05, summarization_temperature: float = 0.2) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._classification_temperature = classification_temperature
        self._summarization_temperature = summarization_temperature
        self._max_question_summary_token_length = max_question_summary_token_length

    def classify(self, prompt: str) -> StockDirection: 
        response = self._transformer_request(prompt=prompt)

        if StockDirection.Up.value in response:
            return StockDirection.Up
        elif StockDirection.Down.value in response:
            return StockDirection.Down
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")

    def _transformer_request(self, prompt: str) -> str:
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

def evaluate(label: str, llm_model: LLMModel, datapoints):
    predictions: list[StockDirection] = []
    true_labels: list[StockDirection] = []
    misses = 0

    for datapoint in tqdm(datapoints, desc="Evaluating"):
        try:
            result = llm_model.classify(prompt=datapoint['completion'])
        except Exception as e:
            misses += 1
            print("ERROR", e.args[0])
            continue 
        if result:
            predictions.append(result.value)
            true_labels.append(datapoint['label'])

    print("Prediction Counts: ", Counter(predictions))
            
    print("="*1)
    print("Results for ", label)
    print("="*10)
    print("Misses", misses)
    print("N of ", len(datapoints))
    print("Accuracy Score: ", accuracy_score(y_true=true_labels, y_pred=predictions))
    print("F1 Score: ", f1_score(y_true=true_labels, y_pred=predictions, pos_label='UP'))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=true_labels, y_pred=predictions, labels=["UP", "DOWN"]))


# NUM_EXAMPLES_TO_EVALUATE = 100
# split_name = 'development'
# answer_datapoints = load_dataset(dataset_name, split=f"{split_name}[0:{NUM_EXAMPLES_TO_EVALUATE}]")

# for r in [1,2,4,6,8]:
#     model, tokenizer = load_fine_tuned_model(f'llama2-v2-50examples--r{r}-a8-e1')
#     evaluate(
#         label=f"R{r}", 
#         llm_model = LLMModel(model=model, tokenizer=tokenizer), 
#         datapoints=answer_datapoints
#     )
#     clean_from_gpu(model)
#     clean_from_gpu(tokenizer)

# """
# Results for  R6 (highest accuracy)
# ==========
# Misses 0
# N of  100
# Accuracy Score:  0.55
# F1 Score:  0.36619718309859156
# Confusion Matrix
# [[13 37]
#  [ 8 42]]
# """

NUM_EXAMPLES_TO_EVALUATE = 1000
split_name = 'test'
answer_datapoints = load_dataset(dataset_name, split=f"{split_name}[0:{NUM_EXAMPLES_TO_EVALUATE}]")

n_examples = 100
model, tokenizer = load_fine_tuned_model(f'llama2-v2-{n_examples}examples--r6-a8-e1')

for _ in range(1,6):
    evaluate(
        label=f"fine-tuned r{n_examples}", 
        llm_model = LLMModel(model=model, tokenizer=tokenizer), 
        datapoints=answer_datapoints
    )
clean_from_gpu(model)
clean_from_gpu(tokenizer)


# model, tokenizer = load_base_model(f'meta-llama/Llama-2-7b-chat-hf')
# evaluate(
#     label=f"base model", 
#     llm_model = LLMModel(model=model, tokenizer=tokenizer), 
#     datapoints=answer_datapoints
# )
# clean_from_gpu(model)
# clean_from_gpu(tokenizer)

