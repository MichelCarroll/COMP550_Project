import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import accumulate

from tqdm.auto import tqdm
from pathlib import Path
import json


def load_transcript(path: Path):
    with path.open('r') as transcript_file:
        return json.load(transcript_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info('Initializing')
    transcript_folder = Path.cwd() / 'transcript_data'
    file_count = sum(1 for _ in transcript_folder.glob('[!_]*.json'))  # Avoids filling memory

    logging.info('Loading data')
    with ProcessPoolExecutor() as pool:
        bundle = []
        mapping = tqdm(
            pool.map(load_transcript, transcript_folder.glob('[!_]*.json')),
            total=file_count,
            desc='Parsing transcript files'
        )

        for transcript in mapping:
            mapping.set_postfix_str(f'Processed {transcript["content"][:-4]}')
            bundle.append(transcript)

    logging.info('Sorting data')
    bundle.sort(key=lambda t: t['date'])
    bundle_path = transcript_folder / '_bundle.json'

    logging.info('Saving data')
    with bundle_path.open('w') as bundle_file:
        json.dump(bundle, bundle_file)
