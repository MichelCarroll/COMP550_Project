from collections import Counter
from entities import Transcript, Prediction, Datasets, DataPoint
from data_loading import load_data_splits
from tqdm import tqdm
import json
from utils import llama2_token_length
from dotenv import load_dotenv
load_dotenv()

import requests

NUMBER_STANDARD_DEVIATIONS_THRESHOLD = 2

datasets: Datasets = load_data_splits()

print("Label counts", Counter([d.true_label for d in datasets.development]))

# ACTIVE_MODEL = 'llama2:7b'
ACTIVE_CLASSIFICATION_MODEL = 'llama:7b-fine-tune-v2'
ACTIVE_SUMMARIZATION_MODEL = 'llama2:7b'


class LLMModel:

    def __init__(self, max_question_summary_token_length: int = 100, classification_temperature: float = 0, summarization_temperature: float = 0.2) -> None:
        self._classification_temperature = classification_temperature
        self._summarization_temperature = summarization_temperature
        self._max_question_summary_token_length = max_question_summary_token_length

    def classify(self, text: str) -> Prediction: 
        output = requests.post(
            url='http://localhost:11434/api/generate',
            data=json.dumps({
                "model": ACTIVE_CLASSIFICATION_MODEL,
                "stream": False,
                "options": {
                    "temperature": self._classification_temperature,
                    "num_predict": 10,
                    "stop": ["\n"],
                },
                "template": "{{ .Prompt }}",
                "prompt": f"""
    [INST] <<SYS>>You are a financial analyst, determining what effect an earnings call will have on the company's stock price, based on a summary of one. Be as critical and skeptical as possible, and remember that the market may disproportionally react to small unexpected details. Respond with one of: {Prediction.Up.value}, {Prediction.Down.value}, {Prediction.Same.value}<</SYS>> {text}[/INST]
    Prediction (one of UP, DOWN, SAME): """})
        )

        response = output.json()['response']
        print(response)
        if Prediction.Up.value in response:
            return Prediction.Up
        elif Prediction.Down.value in response:
            return Prediction.Down
        elif Prediction.Same.value in response:
            return Prediction.Same
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")

    def summarize_into_one_sentence(self, text: str): 
        output = requests.post(
            url='http://localhost:11434/api/generate',
            data=json.dumps({
                "model": ACTIVE_SUMMARIZATION_MODEL,
                "stream": False,
                "options": {
                    "temperature": self._summarization_temperature,
                    "num_predict": self._max_question_summary_token_length,
                    "stop": [],
                },
                "template": "{{ .Prompt }}",
                "prompt": f"""[INST] <<SYS>>You are a financial analyst, determining what effect a portion of an earnings call transcript will have on the company's stock price. Summarize in one brief sentence how this portion will affect the stock prices.<</SYS>> {text}[/INST] Sentence: """})
        )
        return output.json()['response']


def filter_answer(answer: str, token_length_low_threshold: int = 20, token_length_high_threshold: int = 1000) -> bool:
    text_token_length = llama2_token_length(answer)
    return text_token_length >= token_length_low_threshold and text_token_length <= token_length_high_threshold


def llm_classifier_with_metasummary(
        transcript: Transcript, 
        llm_model: LLMModel, 
        verbose: bool = False,
        max_questions_to_summary: int = 20, 
        answer_token_length_low_threshold: int = 20, 
        answer_token_length_high_threshold: int = 1000
    ) -> Prediction:

    answer_texts: list[str] = [
        t for t in transcript.answer_texts() 
        if filter_answer(answer=t, token_length_low_threshold=answer_token_length_low_threshold, token_length_high_threshold=answer_token_length_high_threshold)
    ]
    
    question_summaries = []    
    for answer in tqdm(answer_texts[:max_questions_to_summary], desc="Summarizing answers into one sentence", disable=not verbose):
        summary = llm_model.summarize_into_one_sentence(answer)
        question_summaries.append(summary)

    super_summary = '\n'.join(question_summaries)

    if verbose:
        print("Evaluating summaries and classifying...")
    result = llm_model.classify(super_summary)
    return result


def llm_classifier_with_averaging(
        transcript: Transcript, 
        llm_model: LLMModel, 
        verbose: bool = False,
        max_questions_to_summary: int = 20, 
        answer_token_length_low_threshold: int = 20, 
        answer_token_length_high_threshold: int = 1000
    ) -> Prediction:

    answer_texts: list[str] = [
        t for t in transcript.answer_texts() 
        if filter_answer(answer=t, token_length_low_threshold=answer_token_length_low_threshold, token_length_high_threshold=answer_token_length_high_threshold)
    ]
    
    results: list[int] = []
    for answer in tqdm(answer_texts[:max_questions_to_summary], desc="Summarizing answers into one sentence", disable=not verbose):
        try:
            result = llm_model.classify(answer)
        except:
            continue 
        if result == Prediction.Up:
            results.append(1)
        elif result == Prediction.Down:
            results.append(-1)
        else:
            results.append(0)

    average_result = sum(results) / len(results)

    print("Results: ", results)
    print("Average result: ", average_result)        

    answer = Prediction.Same
    if average_result > 1/3:
        answer = Prediction.Up
    elif average_result < -1/3:
        answer = Prediction.Down

    return answer
    

def evaluate(datapoint: DataPoint) -> bool:
    model = LLMModel()
    try:
        result = llm_classifier_with_averaging(llm_model=model, transcript=datapoint.transcript)
        # result = llm_classifier_with_metasummary(llm_model=model, transcript=datapoint.transcript)
    except Exception as e:
        print("Error occured: ", e)
        return False 
    print(f"Prediction: {result} Actual: {datapoint.true_label}")
    return result == datapoint.true_label

number_of_transcripts_to_evaluate = 10
transcripts_to_evaluate = datasets.development[0:number_of_transcripts_to_evaluate]

results = [
    evaluate(datapoint=datapoint) 
    for datapoint in tqdm(transcripts_to_evaluate, desc="Evaluating transcripts")
]

correct_predictions = sum(results)
total_predictions = len(results)
accuracy = correct_predictions / total_predictions

print("Correct Predictions", correct_predictions)
print("Total Predictions", total_predictions)
print("Accuracy", accuracy)