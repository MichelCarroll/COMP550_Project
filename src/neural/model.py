import torch
from torch import nn
from torch.nn import functional as F  # noqa
from transformers import BertModel, BertTokenizerFast


class NeuralNet(nn.Module):
    def __init__(
            self,
            fin_bert_weights='ProsusAI/finbert',
            lstm_layers=2,
            lstm_hidden_size=128,
            num_classes=2,
            apply_softmax=True,
    ):
        super().__init__()

        # Config
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_output_size = 2 * lstm_layers * lstm_hidden_size
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax

        # Parameters
        # zero initial hidden state is standard
        self.h_0 = nn.Parameter(torch.zeros((2 * lstm_layers, lstm_hidden_size)))
        self.c_0 = nn.Parameter(torch.zeros((2 * lstm_layers, lstm_hidden_size)))

        # Layers
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(fin_bert_weights)
        self.fin_bert = BertModel.from_pretrained(fin_bert_weights)
        self.lstm = nn.LSTM(
            input_size=self.fin_bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
        )
        self.classifier = nn.Linear(self.lstm_output_size, num_classes)

    def tokenize(self, *args, **kwargs):
        return self.bert_tokenizer.tokenize(*args, **kwargs)

    def _encode_transcript(self, transcript: list[str]) -> torch.Tensor:
        """
        Encode the transcript to the embedding space of the model
        :param transcript: The transcript to encode as a list of paragraphs
        :return: Transcript embedding, size lstm_output_size
        """
        input_seq = self.bert_tokenizer(transcript, return_tensors="pt", padding=True)
        output = self.fin_bert(**input_seq)
        cls_tokens = output.last_hidden_state[:, 0]  # sequence length (paras) x hidden size
        lstm_input = cls_tokens.flip(0)  # Invert sequence to perform bottom-up then to-down LSTM
        _, (h_n, _) = self.lstm(lstm_input, (self.h_0, self.c_0))
        return h_n.view(-1)

    def forward(self, batch: list[list[str]]) -> torch.Tensor:
        """
        Forward pass of the Model
        :param batch: input as (BatchSize, Paras)
        :return: output as (BatchSize)
        """

        batch = torch.stack([self._encode_transcript(t) for t in batch])  # Batch Size x lstm_output_size
        output = self.classifier(batch)

        if self.apply_softmax:
            output = F.softmax(output, dim=1)

        return output
