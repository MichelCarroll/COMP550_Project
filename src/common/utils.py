from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def llama2_token_length(text: str) -> int: 
    return len(tokenizer(text)['input_ids'])