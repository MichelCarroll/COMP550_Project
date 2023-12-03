## Loading Dataset

1. Create a `.env` file with an API key from https://eodhd.com/ as the `EOD_API_KEY` environment variable.
2. Change the `YEARS` variable to the range of years you want to load
3. Run `python load_transcript_data.py`

## Evaluating an OpenAI model 

Requirements:
- `OPENAI_KEY` and `HUGGINGFACE_TOKEN` environment variables required

Run `python src/run_openai_classifier.py`

## Evaluating a transformer model 

Requirements:
- `HUGGINGFACE_TOKEN` environment variable required
- GPU instance required (T4 or more powerful)

Run `python src/fine_tune_transformer_model.py`

## Fine-tuning a transformer model

Requirements:
- `HUGGINGFACE_TOKEN` environment variable required
- GPU instance required (T4 or more powerful)

Run `python src/run_transformer_classifier.py`
