import json
import logging
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from tqdm import tqdm

if __name__ == '__main__':
    folder = Path.cwd() / 'transcript_data'

    with open(folder / '_bundle.json') as transcripts:
        transcripts = json.load(transcripts)

    with ZipFile('TMF_dataset.zip', 'w', compression=ZIP_DEFLATED) as myzip:
        myzip.write(folder / '_bundle.json', '_bundle.json')

        logging.info('Zipping text files')
        for transcript in tqdm(transcripts):
            myzip.write(folder / transcript['content'], transcript['content'])
