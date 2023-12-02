import json
import torch
from datasets import Dataset
import gc 
import os 
from dotenv import load_dotenv
import huggingface_hub
from pydantic import BaseModel

load_dotenv()

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

class FineTuningParameters(BaseModel):
    adapter_id_prefix: str
    base_model_name: str
    lora_rank: int = 2
    lora_alpha: int  = 8
    epochs: int = 1
    start_from_checkmarks: bool = False

    def checkmark_dir(self):
        return f"{self.adapter_id_prefix}-{self.lora_rank}-{self.lora_alpha}"

    def adapter_id(self):
        return f"{self.adapter_id_prefix}-r{self.lora_rank}-a{self.lora_alpha}-e{self.epochs}"
    

DATASET_TEXT_FIELD = "completion"
DEVICE_MAP = {"": 0}

def login_to_hub():
    huggingface_hub.login(token=HUGGINGFACE_TOKEN)

def run_model(model, tokenizer, max_new_tokens: int, prompt: str, return_full_text: bool) -> str:
    from transformers import pipeline

    generator = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=max_new_tokens
    )
    return generator(prompt, return_full_text=return_full_text)[0]['generated_text']
    
def load_fine_tuned_model(fine_tuned_adapter_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    fine_tuned_model_file = f"fine_tuned_models/{fine_tuned_adapter_id}"
    with open(f"{fine_tuned_model_file}/adapter_config.json", 'r') as f:
        config_data = json.loads(f.read())
        
    base_model_name = config_data['base_model_name_or_path']

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP
    )
    model = PeftModel.from_pretrained(base_model, fine_tuned_model_file)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def load_base_model(base_model_name: str):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer

    # Reload model in FP16 and merge it with LoRA weights
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=DEVICE_MAP,
    )
    
    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def train_and_save_fine_tuned_model(dataset: Dataset, fine_tuning_parameters: FineTuningParameters):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig
    from trl import SFTTrainer
    from time import time
    
    start = time()

    fine_tuned_model_file = f"fine_tuned_models/{fine_tuning_parameters.adapter_id()}"
    if os.path.exists(fine_tuned_model_file):
        print(f"Model {fine_tuned_model_file} already exists. Skipping")
        return 

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = fine_tuning_parameters.lora_rank

    # Alpha parameter for LoRA scaling
    lora_alpha = fine_tuning_parameters.lora_alpha

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = f"./_model_scratch_space/{fine_tuning_parameters.checkmark_dir()}"

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 25

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        fine_tuning_parameters.base_model_name,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(fine_tuning_parameters.base_model_name, trust_remote_code=True, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=fine_tuning_parameters.epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=DATASET_TEXT_FIELD,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train(resume_from_checkpoint=fine_tuning_parameters.start_from_checkmarks)

    trainer.model.save_pretrained(fine_tuned_model_file)

    clean_from_gpu(base_model)
    clean_from_gpu(trainer)
    
    training_duration = f"{time() - start} seconds"
    with open(f"{fine_tuned_model_file}/metadata.txt", 'w') as f:
        f.write(f"Parameters: {fine_tuning_parameters.json()}\n")
        f.write(f"Duration: {training_duration}\n")
        f.write(json.dumps(trainer.state.log_history))

def clean_from_gpu(obj):
    del obj
    gc.collect()
    torch.cuda.empty_cache()