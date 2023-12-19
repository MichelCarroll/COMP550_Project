import json
import re
import zipfile
from bisect import bisect_left, bisect_right
from collections import UserDict
from dataclasses import dataclass
from datetime import datetime
from functools import total_ordering
from pathlib import Path
from typing import overload, Iterable


@dataclass
@total_ordering
class Quarter:
    quarter: int
    year: int

    def __init__(self, string):
        self.quarter, self.year = re.search(r'Q([1-4]) ([0-9]{4})', string).groups()

    def __str__(self):
        return f'Q{self.quarter} {self.year}'

    def __repr__(self):
        return f'<Quarter {self!s}>'

    def __lt__(self, other: 'Quarter'):
        if self.year < other.year:
            return True
        if self.year == other.year:
            return self.quarter < other.quarter
        return False

    def __eq__(self, other: 'Quarter'):
        return self.year == other.year and self.quarter == other.quarter


class TranscriptView(UserDict):
    def __init__(self, mapping, dataset_path):
        super().__init__(mapping)
        self.__dataset_path = dataset_path

    def __getitem__(self, item):
        if item == 'content':
            return self._load_content()
        return self.data[item]

    def _load_content(self):
        content_path = self.__dataset_path / self.data['content']
        with content_path.open('r', encoding='utf8') as content_file:
            return content_file.read()


class MotleyFoolDataset:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise ValueError(f'Dataset does not exist at {dataset_path}')

        if self.dataset_path.suffix == '.zip':
            self.dataset_path = zipfile.Path(self.dataset_path)

        bundle_path = self.dataset_path / '_bundle.json'
        with (bundle_path.open() as bundle):
            self.metadata = [{
                **instance,
                'quarter': Quarter(instance['quarter']),
                'date': datetime.fromisoformat(instance['date']),
            } for instance in json.load(bundle)]
            self.metadata.sort(key=lambda e: e['quarter'])

    def __len__(self):
        return len(self.metadata)

    @overload
    def __getitem__(self, item: int) -> dict:
        ...

    @overload
    def __getitem__(self, item: slice | str) -> Iterable[dict]:
        ...

    def __getitem__(self, item):
        # If vanilla get item from index
        if isinstance(item, int):
            return self._wraps(self.metadata[item])

        # Select all data from a quarter
        if isinstance(item, str):
            return self._search(item, item)

        # Rest of logic is only for slices
        if not isinstance(item, slice):
            return NotImplemented

        if isinstance(item.start, int) or isinstance(item.stop, int):
            return self._wraps_slice(item)

        if isinstance(item.start, str) or isinstance(item.stop, str):
            return self._search(item.start, item.stop)

        return NotImplemented

    def __iter__(self):
        for instance in self.metadata:
            yield self._wraps(instance)

    def range(self, q_start: str = None, q_stop: str = None):
        """
        Search elements in the given Quarter range, stop is inclusive
        """
        start = bisect_left(self.metadata, Quarter(q_start), key=lambda e: e['quarter']) if q_start else 0
        stop = bisect_right(self.metadata, Quarter(q_stop), key=lambda e: e['quarter']) if q_stop else len(self)
        return start, stop

    def _search(self, q_start: str, q_stop: str):
        start, stop = self.range(q_start, q_stop)
        return self._wraps_slice(slice(start, stop))

    def _wraps_slice(self, selector: slice):
        return [self._wraps(instance) for instance in self.metadata[selector]]

    def _wraps(self, instance):
        return TranscriptView(instance, self.dataset_path)
