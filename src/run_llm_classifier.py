from common.entities import StockDirection, Datasets, AnswerDataPoint
from common.data_loading import load_data_splits
from tqdm import tqdm
from common.utils import llama2_token_length
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from random import seed, shuffle
import os 
from common.transformer_operations import run_model, load_fine_tuned_model, load_base_model

load_dotenv()

SEED = os.environ['SEED']
seed(SEED)

datasets: Datasets = load_data_splits()

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
        prompt = f"""[INST] <<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Be as critical and skeptical as possible. Respond with {StockDirection.Up.value} or {StockDirection.Down.value}<</SYS>> {text}[/INST]
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


NUM_EXAMPLES_TO_EVALUATE = 100

shuffle(datasets.development)
answer_datapoints = datasets.development[0:NUM_EXAMPLES_TO_EVALUATE]


# model, tokenizer = load_fine_tuned_model('')
# model, tokenizer = load_base_model('')

evaluate(
    label="Test", 
    llm_model = LLMModel(model=model, tokenizer=tokenizer), 
    datapoints=answer_datapoints
)


# evaluate(
#     label="Base Llama2 13B (Chat)", 
#     llm_model = LLMModel(model='meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d', model_type=LLMModelType.Replicate), 
#     datapoints=answer_datapoints
# )

# evaluate(
#     label="Base Llama2 70B (Chat)", 
#     llm_model = LLMModel(model='meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3', model_type=LLMModelType.Replicate), 
#     datapoints=answer_datapoints
# )

# evaluate(
#     label="Base Llama2 7B", 
#     llm_model = LLMModel(model='llama2:7b', model_type=LLMModelType.Ollama), 
#     datapoints=answer_datapoints
# )

# evaluate(
#     label="Base Mistral (instruct) 7B", 
#     llm_model = LLMModel(model='mistral', model_type=LLMModelType.Ollama), 
#     datapoints=answer_datapoints
# )

# evaluate(
#     label="Fine-Tuned Llama2 7B (1000 examples)", 
#     llm_model = LLMModel(model='llama:7b-fine-tune-v4', model_type=LLMModelType.Ollama), 
#     datapoints=answer_datapoints
# )
