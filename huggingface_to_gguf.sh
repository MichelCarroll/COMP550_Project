
# https://github.com/ggerganov/llama.cpp/discussions/2948

MODEL_FOLDER=llama2-7b-earnings-stock-prediction-fine-tune-1000-examples-binary-v2

cp tokenizer.model $MODEL_FOLDER/

python llama.cpp/convert.py $MODEL_FOLDER \
  --outfile $MODEL_FOLDER.gguf \
  --outtype q8_0