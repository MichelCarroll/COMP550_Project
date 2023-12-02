import os
import huggingface_hub
from datasets import load_dataset
import os 
from dotenv import load_dotenv

from common.transformer_operations import train_and_save_fine_tuned_model, FineTuningParameters

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

base_model_name = 'meta-llama/Llama-2-7b-chat-hf'
adapter_id_prefix = 'llama2-1000examples-'
dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-v2'
n_examples_to_train = 1000

huggingface_hub.login(token=HUGGINGFACE_TOKEN)
dataset = load_dataset(dataset_name, split=f"train[0:{n_examples_to_train}]")

# from collections import Counter
# print(Counter(dataset['label']))
# > Counter({'UP': 500, 'DOWN': 500})


for r in [2,4,6,8,10]:
    train_and_save_fine_tuned_model(
        dataset=dataset,
        fine_tuning_parameters=FineTuningParameters(
            adapter_id_prefix=adapter_id_prefix,
            base_model_name=base_model_name,
            lora_rank=r,
            lora_alpha=8,
            epochs=1
        )
    )

    