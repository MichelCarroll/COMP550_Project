This is the code used for *Comparative Analysis on Stock Price Direction using Earnings Call Transcripts* project for McGill COMP 550.

Authors:
- Michel Carroll (michel.carroll@mail.mcgill.ca)
- Damien Djomby (christopher.salomons@mail.mcgill.ca)
- Christopher Salomons (damien.djomby@mail.mcgill.ca)

# Sample Data

- `sample-scraped-full-transcript.json`: Sample metadata for our full text dataset
- `sample-scraped-full-transcript.txt`: Sample full transcript text
- `sample-scraped-text-block-transcript.json`: Sample text block extraction

## Prompt template for the Llama2 model

[INST] <\<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Respond with UP or DOWN<\</SYS>> {text}[/INST]
Direction (UP or DOWN):

## Full sample completion (Llama2 model)

[INST] <\<SYS>>You are a financial analyst, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Respond with UP or DOWN<\</SYS>> Well, I wouldn't say necessarily front-end loaded. I mean, we just printed Q1 at $300 million -- approximately $380 million acquired. And so I think we can cut that up for the remaining three quarters. I don't anticipate Q2 to be very different in terms of assets that we're acquiring. Volume may be down or at that level. So we'll see where things close. Again, we can't predict whether something is going to close on July 27, 28th or July 2. It won't close on July four because everything is closed. But our business, the last outstanding issue generally to close an asset in our business is reliant upon the tenant to provide estoppels. And so things can cross quarters.\nWe have ideas that will close this quarter and then everything can get jumbled around. ARC gives us the visibility to move things around in those corners as we have those third-party respondents, diligence, outstanding estoppels and such like that. And so that provides a level of transparency, visibility for us. But closing and the timing of transactions really isn't relying necessarily upon just our operations or execution here.[/INST]
Direction (UP or DOWN): DOWN

## Prompt for GPT-4

system message=You are a binary classifier with expert financial analyst knowledge, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Output either UP if you predict the stock will go up, or DOWN if you predict it will go down. You must absolutely make a prediction – don't answer with N/A.

user message=The answer from the earnings transcript is: {text}

Output function call JSON schema: 
```
{
    "name": "predict",
    "description": "Label the correct class",
    "parameters": {
        "type": "object",
        "properties": {
            'prediction': {
                "type": "string",
                "enum": ["UP", "DOWN"]
            },
        },
        "required": [ "prediction"]
    }
}
```

temperature: 0.2

## Sample transcript metadata

```
{
    "company_name":"Barracuda Networks",
    "company_ticker":"CUDA",
    "quarter":"Q2",
    "date":"2017-10-10T16:30:00Z",
    "content":"2017-10-11-barracuda-networks-q2-2018-earnings-conference-cal.txt",
    "daily_volatility":0.022984554070146792,
    "closing_price_day_before":["2017-10-09",25.73],
    "closing_price_day_of":["2017-10-10",25.74],
    "closing_price_day_after":["2017-10-11",22.65]
}
```

# Usage 

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
