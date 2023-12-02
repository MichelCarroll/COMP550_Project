import os
import huggingface_hub
from datasets import load_dataset
import os 
from dotenv import load_dotenv

from common.transformer_operations import train_and_save_fine_tuned_model, FineTuningParameters

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

base_model_name = 'meta-llama/Llama-2-7b-chat-hf'
adapter_id_prefix = 'llama2-experiment'
dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-v2'

huggingface_hub.login(token=HUGGINGFACE_TOKEN)
dataset = load_dataset(dataset_name, split="train[0:1000]")

train_and_save_fine_tuned_model(
    dataset=dataset,
    fine_tuning_parameters=FineTuningParameters(
        adapter_id_prefix=adapter_id_prefix, 
        base_model_name=base_model_name        
    )
)
