import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding
from tqdm.auto import tqdm


class TMFDatasetAdapter(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizerBase, device: torch.device = None):
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device
        self.data = [
            (self._preprocess_data(instance['content']), 1 if instance['direction'] == 'UP' else 0)
            for instance in tqdm(data)
        ]

    def _preprocess_data(self, content) -> BatchEncoding:
        stripped_content = [
            para.strip().replace(' -- ', ', ')  # Replace speakers' double dash by a simple coma for simplification
            # Split text on double line ending to detect paragraphs, particularity of our data
            for para in content.split('\n\n')
            # Drop short para, mostly "Alright, thank you <somebody>" or sections titles that the model would infer from
            # the content of the sentence. However, we still want to know when the operator is speaking
            if len(para) > 25 or "operator" in para.lower()
        ]

        encodings = self.tokenizer(stripped_content, return_tensors="pt", padding=True, truncation=True)

        if self.device:
            encodings = encodings.to(self.device)

        return encodings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)
