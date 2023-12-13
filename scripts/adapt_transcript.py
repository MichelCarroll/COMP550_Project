import json
import re
from pathlib import Path

from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging


def adapt_transcript(path: Path):
    with path.open('r') as transcript_file:
        transcript = json.load(transcript_file)

    content_path = path.parent / (path.stem + '.txt')

    if not transcript['content'].endswith('.txt'):
        with content_path.open('wb') as content_file:
            content_file.write(transcript['content'].encode())
            transcript['content'] = content_path.name

    if re.match(r'Q[1-4]', transcript['quarter']):
        q, a = re.search(r'q([1-4])-([0-9]{4})', path.stem).groups()
        transcript['quarter'] = f'Q{q} {a}'

    with path.open('w') as transcript_file:
        json.dump(transcript, transcript_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    transcript_folder = Path.cwd() / 'transcript_data'
    logging.info("Looking in folder %s", transcript_folder)

    logging.info('Initializing')
    file_count = sum(1 for _ in transcript_folder.glob('[!_]*.json'))  # Avoids filling memory

    logging.info('Processing data')
    with ProcessPoolExecutor() as pool:
        for _ in tqdm(map(adapt_transcript, transcript_folder.glob('[!_]*.json')), total=file_count):
            pass
