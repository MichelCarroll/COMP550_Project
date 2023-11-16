from transformers import LlamaTokenizer
from entities import Transcript, Prediction
from evaluation import true_stock_direction
import os 
from tqdm import tqdm
import pytz
import json
from dotenv import load_dotenv
load_dotenv()

import requests

utc = pytz.UTC
 
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

TRANSCRIPTS_DATA_PATH = 'transcript_data'
DEVELOPMENT_SET_SPLIT_RATIO = 0.2
NUMBER_STANDARD_DEVIATIONS_THRESHOLD = 3

def load_all_transcripts() -> list[Transcript]:
    all_transcripts: list[Transcript] = []
    for file in tqdm(os.listdir(TRANSCRIPTS_DATA_PATH), desc="Loading transcripts into memory"):
        with open(f"{TRANSCRIPTS_DATA_PATH}/{file}") as f:
            if not file.endswith('.json'):
                continue 
            all_transcripts.append(Transcript.model_validate_json(f.read()))
    return all_transcripts

transcripts = load_all_transcripts()
transcripts.sort(key=lambda t: t.event_time.replace(tzinfo=utc))

split_n = int(DEVELOPMENT_SET_SPLIT_RATIO * len(transcripts))
development_set_transcripts: list[Transcript] = transcripts[:split_n]
test_set_transcripts: list[Transcript] = transcripts[split_n:]

development_set_labels: list[Prediction] = [
    true_stock_direction(transcript=t, standard_deviation_multiples=NUMBER_STANDARD_DEVIATIONS_THRESHOLD) 
    for t in development_set_transcripts
]

# from collections import Counter
# print("Label counts", Counter(development_set_labels))

first_transcript = development_set_transcripts[0]

ACTIVE_MODEL = 'llama2:7b'

class LLMModel:

    def __init__(self, model_name: str, max_question_summary_token_length: int = 100, classification_temperature: float = 0, summarization_temperature: float = 0.2) -> None:
        self._model_name = model_name
        self._classification_temperature = classification_temperature
        self._summarization_temperature = summarization_temperature
        self._max_question_summary_token_length = max_question_summary_token_length

    def classify(self, text: str) -> Prediction: 
        output = requests.post(
            url='http://localhost:11434/api/generate',
            data=json.dumps({
                "model": self._model_name,
                "stream": False,
                "options": {
                    "temperature": self._classification_temperature,
                    "num_predict": 10,
                    "stop": ["\n"],
                },
                "template": "{{ .Prompt }}",
                "prompt": f"""
    [INST] <<SYS>>You are a financial analyst, determining what effect an earnings call will have on the company's stock price, based on a summary of one. Be as critical and skeptical as possible, and remember that the market may disproportionally react to small unexpected details. Respond with one of: {Prediction.Up.value}, {Prediction.Down.value}, {Prediction.Same.value}<</SYS>> {text}[/INST]
    Prediction (one of UP, DOWN, SAME): {Prediction.Up.value}
    [INST]The company has run into a lot of issues this year.[/INST]
    Prediction (one of UP, DOWN, SAME): {Prediction.Down.value}
    [INST]The company will continue its steady course.[/INST]
    Prediction (one of UP, DOWN, SAME): {Prediction.Same.value}
    [INST]{text}[/INST]
    Prediction: """})
        )

        response = output.json()['response']
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
                "model": self._model_name,
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

def llm_classifier(
        transcript: Transcript, 
        llm_model: LLMModel, 
        verbose: bool = False,
        max_questions_to_summary: int = 20, 
        answer_token_length_low_threshold: int = 20, 
        answer_token_length_high_threshold: int = 1000
    ) -> Prediction:

    answer_texts: list[str] = []
    last_block_was_analyst: bool = False

    for block in transcript.text_blocks:
        if block.section == 'Questions and Answers':
            if 'Operator' in block.speaker:
                continue 
            elif last_block_was_analyst:
                text_token_length = len(tokenizer(block.text)['input_ids'])
                if text_token_length >= answer_token_length_low_threshold and text_token_length <= answer_token_length_high_threshold:
                    answer_texts.append(block.text)
                last_block_was_analyst = False 
            elif 'Analyst' in block.speaker:
                last_block_was_analyst = True

    question_summaries = []    
    for answer in tqdm(answer_texts[:max_questions_to_summary], desc="Summarizing answers into one sentence", disable=not verbose):
        summary = llm_model.summarize_into_one_sentence(answer)
        question_summaries.append(summary)

    super_summary = '\n'.join(question_summaries)

    if verbose:
        print("Evaluating summaries and classifying...")
    result = llm_model.classify(super_summary)
    return result

def evaluate(transcript: Transcript, true_label: Prediction) -> bool:
    model = LLMModel(model_name=ACTIVE_MODEL)
    try:
        result = llm_classifier(llm_model=model, transcript=transcript)
    except Exception as e:
        print("Error occured: ", e)
        return False 
    print(f"Prediction: {result} Actual: {true_label}")
    return result == true_label

number_of_transcripts_to_evaluate = 10
transcripts_to_evaluate = list(zip(development_set_transcripts, development_set_labels))[0:number_of_transcripts_to_evaluate]

results = [
    evaluate(transcript=transcript, true_label=true_label) 
    for transcript, true_label in tqdm(transcripts_to_evaluate, desc="Evaluating transcripts")
]

correct_predictions = sum(results)
total_predictions = len(results)
accuracy = correct_predictions / total_predictions

print("Correct Predictions", correct_predictions)
print("Total Predictions", total_predictions)
print("Accuracy", accuracy)