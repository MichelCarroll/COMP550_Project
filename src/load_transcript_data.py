import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from typing import List, Optional, Tuple
from datetime import date, timedelta
from dateutil.parser import parse as parse_datetime
from tenacity import retry, stop_after_attempt, wait_random_exponential

import requests
import os
from dotenv import load_dotenv

load_dotenv()

from entities import TextBlock, Transcript

YEARS = [2021]

EOD_API_KEY = os.environ['EOD_API_KEY']


def transcript_urls_from_sitemap(year: int, month: int) -> list[str]:
    sitmap_url = f'https://www.fool.com/sitemap/{year}/{str(month).zfill(2)}'
    print(f"Pulling sitemap {sitmap_url}...")
    sitemap_content = requests.get(sitmap_url).text
    soup = BeautifulSoup(sitemap_content, 'lxml')
    earning_transcript_links = [url.text for url in soup.find_all('loc') if '/earnings/call-transcripts/' in url.text]
    return earning_transcript_links

def all_transcript_urls_from_sitemap(years: list[int]) -> list[str]:
    all_transcript_urls = [
        l 
        for year in years 
        for month in range(1,13) 
        for l in transcript_urls_from_sitemap(year=year, month=month)
    ]
    return all_transcript_urls

@retry(
    wait=wait_random_exponential(min=5, max=20),
    stop=stop_after_attempt(3)
)
def ticker_eod_data(ticker: str) -> dict:
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}.US?api_token={EOD_API_KEY}&fmt=json"
    response = requests.request("GET", url, headers={}, data={})
    return response.json()

def cached_ticker_eod_data(ticker: str) -> dict:
    file_path = f'eod_data_cache/{ticker}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.loads(f.read())
        
    data = ticker_eod_data(ticker=ticker)
    with open(file_path, 'w') as f:
        f.write(json.dumps(data))

    return data

@retry(
    wait=wait_random_exponential(min=5, max=20),
    stop=stop_after_attempt(3)
)
def transcript_from_url(url: str) -> Transcript:
    transcript = Transcript(url=url)

    transcript_html = requests.get(transcript.url).text    
    soup = BeautifulSoup(transcript_html, 'lxml')

    event_date = soup.find(class_='tailwind-article-body').find(id='date').text
    event_time = soup.find(class_='tailwind-article-body').find(id='time').text

    event_time = event_time.replace('ET', 'EST')

    transcript.event_time = parse_datetime(f"{event_date} {event_time}")
    transcript.headline = soup.find(itemprop='name')['content']
    transcript.company_name = re.search(r"(.*)\(+", transcript.headline).group(1)
    transcript.company_ticker = re.search(r"\((.*)\)", transcript.headline).group(1)

    quarter_result = re.search(r"\)\ (.*)\ Earnings", transcript.headline)
    transcript.fiscal_quarter = quarter_result.group(1) if quarter_result else None
    transcript.fiscal_quarter

    text_blocks = []

    for i, section in enumerate([t.text for t in soup.find(class_='tailwind-article-body').find_all('h2')]):
        if 'Prepared Remarks' in section:
            section = 'Prepared Remarks'
        elif 'Questions' in section:
            section = 'Questions and Answers'
        else:
            section = 'Other'
            
        current_text_block = TextBlock(section=section, speaker='N/A', text='')

        for sibling in soup.find(class_='tailwind-article-body').find_all('h2')[i].find_next_siblings():
            if sibling.name == 'p':
                if sibling.find("strong"):
                    if current_text_block.text != '':
                        text_blocks.append(current_text_block)
                    current_text_block = TextBlock(section=section, speaker=sibling.text, text='')
                else:
                    current_text_block.text += '\n' + sibling.text
                    
            elif sibling.name == 'h2':
                if current_text_block.text != '':
                    text_blocks.append(current_text_block)
                break

    transcript.text_blocks = text_blocks

    eod_company_data = cached_ticker_eod_data(transcript.company_ticker)
    eod_price_by_date = { d['date']: d['adjusted_close'] for d in eod_company_data }

    df = pd.DataFrame(eod_company_data)
    df['date'] = pd.to_datetime(df.date)
    df = df[df.date.dt.year == transcript.event_time.year]
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_volatility = df.daily_return.dropna().std()
    transcript.daily_volatility = daily_volatility

    def next_price(price_date: date, going_backwards: bool, n: int = 1) -> Optional[Tuple[date, float]]:
        price_date=price_date - timedelta(days=1) if going_backwards else price_date + timedelta(days=1)
        if n > 5:
            return None
        value = eod_price_by_date.get(str(price_date))
        if value:
            return (price_date, value)
        else:
            return next_price(
                price_date=price_date, 
                going_backwards=going_backwards,
                n=n+1
            )

    if transcript.event_time:
        event_date = transcript.event_time.date()
        transcript.closing_price_day_before = next_price(price_date=event_date, going_backwards=True)
        transcript.closing_price_day_of = (event_date, eod_price_by_date.get(str(event_date)))
        transcript.closing_price_day_after = next_price(price_date=event_date, going_backwards=False)
        
    return transcript

def clean_filename(url: str) -> str:
    segment = url.strip('/').split('/')[-1]
    cleaned_segment = re.sub(r'[^a-zA-Z0-9\-]', '', segment)
    return cleaned_segment


def save_transcript_from_url(url: str, skip_if_exists: bool = True):
    file_path = f'transcript_data/{clean_filename(url)}.json'
    if skip_if_exists and os.path.exists(file_path):
        print("Skipping because exists")
        return 
    
    transcript = transcript_from_url(url=url)
    with open(file_path, 'w') as f:
        f.write(transcript.model_dump_json())

print("Pulling transcript URLs from sitemap...")
transcript_urls = all_transcript_urls_from_sitemap(years=YEARS)
print(f"Processing total of {len(transcript_urls)} transcripts")
for transcript_url in transcript_urls:
    print(f"Processing {transcript_url}...")
    try:
        save_transcript_from_url(url=transcript_url)
    except Exception as e:
        print(f"Could not process. {e.args}")