import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Iterable, Sequence

import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse, ParserError
from tqdm.auto import tqdm

XML_NS = {'sitemap': 'http://www.google.com/schemas/sitemap/0.84'}
URL_PREFIX = "https://www.fool.com/earnings/call-transcripts/"
ERR_DUMPS = Path.cwd() / 'err_dumps'


def parse_date(text):
    corrected = re.sub(r'(\D)([0-9])(\d\d)(\D)', r'\1\2:\3\4', text)
    corrected = re.sub(r'(\D)([0-9])\s(\d\d)(\D)', r'\1\2:\3\4', corrected)
    corrected = corrected.replace('pam', 'am')
    return parse(corrected, tzinfos={'ET': 'EST', 'EDT': 'EST', 'CT': 'CST'})


class Scraper:
    """
    a Motley Fool scraper

    We use a ``requests.session`` to keep a persistent connection to `Motley Fool's website <https://www.fool.com/>`_,
    which drastically speedup the scrapping.
    """
    def __init__(self, destination: Path, errors_file: Path, errors_dumps: Path):
        """
        Create a Motley Fool scraper
        :param destination: Folder in which every transcript will be saved
        :param errors_file: File in which the URLs of transcripts couldn't be parsed will be saved
        :param errors_dumps: A folder in which will be dumped the html content of transcript that couldn't be parsed
        """
        self._session = requests.session()
        self._destination = destination
        self._errors_file = errors_file
        self._errors_dumps = errors_dumps

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._session.close()
        return False

    def parse_sitemap(self, sitemap_url):
        sitemap = self._session.get(sitemap_url)

        try:
            urlset = ET.fromstring(sitemap.text)
        except ET.ParseError:
            logging.exception('Unable to parse sitemap @ %s', sitemap_url)
            logging.error("Sitemap content:\n%s", sitemap.text)
            raise

        for url in urlset:
            loc = url.findtext('sitemap:loc', namespaces=XML_NS)
            if '/earnings/call-transcripts/' in loc:
                yield loc

    @staticmethod
    def get_sitemap_urls(years: Iterable[int]) -> Iterator[str]:
        # Minimum year is 2017, first year to present earning calls

        return (
            f'https://www.fool.com/sitemap/{year}/{str(month).zfill(2)}'
            for year in years
            for month in range(1, 13)
        )

    def scrap_transcript(self, url):
        # Download and parse
        transcript_html = self._session.get(url).text
        soup = BeautifulSoup(transcript_html, 'lxml')
        transcript_body = soup.find(class_='tailwind-article-body')

        # === Preprocess cleanup ===
        # A link to the ticker may be present, extracting it before removing links
        ticker_symbol = transcript_body.find('a', class_="ticker-symbol", recursive=True)

        # Removes Watermark if present
        if watermark := transcript_body.find(class_='image imgR'):
            watermark.decompose()

        # Remove disclaimer at the end of the article
        terms = transcript_body.find('a', string='Terms and Conditions')
        if terms is not None:
            disclaimer = terms.parent
            disclaimer.decompose()

        # Removes useless tags for resulting transcript and simplify parsing
        # The following will also simplify mentions of participants
        for tag in transcript_body.find_all(['strong', 'em', 'span']):
            tag.unwrap()

        # Remove added links at the bottom of the page
        for link in transcript_body.find_all('a'):
            parent = link.parent
            link.decompose()
            if not parent.contents:
                parent.decompose()

        # Remove ads from Motley Fool
        for ad in transcript_body.find_all(class_='article-pitch-container'):
            ad.decompose()

        transcript_body.smooth()

        # === Extracting information ===
        title = soup.title.text
        company_name = re.search(r'^(.*?) (?:\(|Q[1-4])', title).group(1)

        # First paragraph is company's information
        meta = transcript_body.find('p', recursive=False)

        if ticker_symbol is not None:
            company_ticker = ticker_symbol.text
        else:
            # If company ticker has yet to be found, searching it in the title
            company_ticker_reg = re.search(r'\((?:(?:NYSE|NASDAQ):\s?)?([A-Z\d]+[-.: ]?[A-Z\d]*)\)', title)
            # If the company is not found in the title, try extracting it from the information at the top of the page
            if company_ticker_reg is None:
                company_ticker_reg = re.search(r'\((?:(?:NYSE|NASDAQ):\s?)?([A-Z\d]+[-.: ]?[A-Z\d]*)\)',
                                               meta.get_text())
            company_ticker = company_ticker_reg.group(1)

        # Date is inlined date thanks to previous cleanup
        try:
            event_dt = parse_date(list(meta.strings)[-1])
        except ParserError:
            # Sometimes the date is in the second paragraph, the 1st being the name of the company
            meta, name = meta.findNextSibling('p', recursive=False), meta
            name.decompose()
            event_dt = parse_date(list(meta.strings)[-1])

        # Look for the quarter in the title
        quarter_reg = re.search(r'(Q[1-4]|FY)\s20[0-9]{2}', title)
        # If the quarter cannot be extracted from the title, it may not contain the year.
        # We try to capture the quarter form the first paragraph, even though it is more random
        if quarter_reg is None:
            quarter_reg = re.search(r'(Q[1-4]|H[1-2]|FY)\s20[0-9]{2}', meta.get_text())
        if quarter_reg is not None:  # If it is FINALLY not None
            quarter = quarter_reg.group()
        # If we still have no quarter, we'll try to deduct it from the date
        else:
            q = re.search(r'Q([1-4]) ', title).group(1)  # Just fail if not present
            # call may be for the same year if consistent with current month, otherwise take previous
            y = event_dt.year if event_dt.month > int(q) * 3 else event_dt.year - 1
            quarter = f'Q{q} {y}'

        # Meta data is no longer required
        meta.decompose()

        return {
            'url': url,
            'title': title,
            'company_name': company_name,
            'company_ticker': company_ticker,
            'quarter': quarter,
            'date': event_dt.isoformat(),
            'content': transcript_body.get_text(separator="\n").strip(),
        }

    def save_transcript(self, url, *, override=False) -> bool:
        """
        Scrape, parse and save an earning call transcript from Motley Fool
        :param url: Motley Fool url
        :param override: Weather or not to replace existing files
        :return: True if the transcript has been processed, False if skipped because already present
        """
        file_prefix = url.removeprefix(URL_PREFIX).removesuffix('.aspx').strip('/').replace('/', '-')
        filename = file_prefix + '.json'
        filepath = self._destination / filename

        if not override and filepath.exists():
            logging.debug('File `%s` already exists, skipping', filename)
            return False

        logging.debug('Scraping `%s`', url)
        transcript = self.scrap_transcript(url)

        content_filename = file_prefix + '.txt'
        content_path = self._destination / content_filename
        content = transcript['content'].encode()

        logging.debug('Saving `%s`', filename)
        with filepath.open('w') as file, content_path.open('wb') as content_file:
            content_file.write(content)
            transcript['content'] = content_filename
            json.dump(transcript, file)

        return True

    def scrape(self, years: Sequence[int]):
        sitemap_cache_path = self._destination / '_sitemap_cache.txt'

        # Create a cache with all urls to be able to estimate time required to scrap
        if not sitemap_cache_path.exists():
            count = 0
            with sitemap_cache_path.open('w') as sitemap_cache:
                logging.info('No cache for sitemap data, downloading sitemaps')
                it = tqdm(self.get_sitemap_urls(years), total=len(years) * 12, desc='Parsing sitemaps')

                for sitemap_url in it:
                    it.set_postfix(url=sitemap_url)
                    for url in self.parse_sitemap(sitemap_url):
                        print(url, file=sitemap_cache)
                        count += 1

            logging.info('Sitemap cache data stored in %s', sitemap_cache_path)

        else:
            logging.info('Sitemap cache data found, counting urls')
            count = sum(1 for _ in sitemap_cache_path.open('r'))
            logging.info('%d urls found!', count)

        with sitemap_cache_path.open('r') as sitemap_cache:
            # No need to parallelize, we'll get IP blacklisted if we are too fast
            skipped_count = 0
            errors = 0
            it = tqdm(sitemap_cache, total=count, desc='Scrapping transcripts')
            for transcript_url in it:
                transcript_url = transcript_url.strip()
                it.set_postfix({
                    'URL': transcript_url,
                    'Skipped': skipped_count,
                    'Errors': errors
                })

                try:
                    skipped = not self.save_transcript(transcript_url)
                    skipped_count += skipped
                except Exception as e:
                    errors += 1
                    with self._errors_file.open('a') as err:
                        print(transcript_url, file=err)

    def retry_errors(self):
        # We
        with open(self._errors_file, 'r') as errors:
            for error in errors:
                print(error)
                break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ERR_DUMPS.mkdir(exist_ok=True)

    YEARS = range(2017, 2024)
    out_folder = Path.cwd() / 'transcript_data'
    out_folder.mkdir(exist_ok=True)

    err_file = ERR_DUMPS / 'errors.txt'

    with Scraper(out_folder, err_file, ERR_DUMPS) as scraper:
        scraper.scrape(YEARS)

