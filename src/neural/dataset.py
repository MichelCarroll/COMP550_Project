from torch.utils.data import Dataset

from src.dataset_utils import MotleyFoolDataset


class TMFDatasetAdapter(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()

        self.data = [
            (self._preprocess_data(instance['content']), instance['direction'])
            for instance in data
        ]

    @staticmethod
    def _preprocess_data(content):
        return [
            para.strip().replace(' -- ', ', ')  # Replace speakers' double dash by a simple coma for simplification
            # Split text on double line ending to detect paragraphs, particularity of our data
            for para in content.split('\n\n')
            # Drop short para, mostly "Alright, thank you <somebody>" or sections titles that the model would infer from
            # the content of the sentence. However, we still want to know when the operator is speaking
            if len(para) > 25 or "operator" in para.lower()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)
