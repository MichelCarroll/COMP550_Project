
# https://github.com/ggerganov/llama.cpp/discussions/2948

cp tokenizer.model llama2-7b-earnings-stock-prediction-fine-tune/

python llama.cpp/convert.py llama2-7b-earnings-stock-prediction-fine-tune \
  --outfile llama2-7b-earnings-stock-prediction-fine-tune.gguf \
  --outtype q8_0