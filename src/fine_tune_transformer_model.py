import os
import huggingface_hub
from datasets import load_dataset
import os 
from dotenv import load_dotenv

from common.transformer_operations import train_and_save_fine_tuned_model, FineTuningParameters

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

base_model_name = 'meta-llama/Llama-2-7b-chat-hf'

dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-binary'


# for n_examples_to_train in [10, 50, 100, 500, 1000]:
for n_examples_to_train in [100, 500, 1000]:
    
    adapter_id_prefix = f'llama2-v2-{n_examples_to_train}examples-'
    huggingface_hub.login(token=HUGGINGFACE_TOKEN)
    dataset = load_dataset(dataset_name, split=f"train[0:{n_examples_to_train}]")

    # for r in [1,2,4,6,8,10]:
    for r in [6]:
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

    