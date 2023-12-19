import argparse
import logging

import torch
from torch.nn import functional as F
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import DefaultDataCollator

from neural.dataset import TMFDatasetAdapter
from dataset_utils import MotleyFoolDataset
from neural.model import NeuralNet


logging.basicConfig(level=logging.INFO)


def collate(batch):
    X_batch = []  # noqa
    y_batch = []

    for example, target in batch:
        X_batch.append(example)
        y_batch.append(target)

    return X_batch, torch.tensor(y_batch)


def train(model: NeuralNet, dataloader: DataLoader, loss_fn, optimizer, device):
    iterator = tqdm(dataloader)
    model.train()

    for batch, (X, y) in enumerate(iterator):
        y = y.to(device)
        X = [x.to(device) for x in X]  # noqa

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            iterator.set_postfix({
                'loss': loss.item()
            })


def test(model: NeuralNet, dataloader: DataLoader, loss_fn, device):
    size = len(dataloader.dataset)  # noqa
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            X = [x.to(device) for x in X]  # noqa

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main(dataset: MotleyFoolDataset, weights, epochs):
    logging.info("Loading model with weights from %s", weights)
    model = NeuralNet(fin_bert_weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Running PyTorch on device: %s", device)

    logging.info("Loading datasets")
    train_set = DataLoader(TMFDatasetAdapter(dataset[:'Q4 2021'][:10], model.bert_tokenizer),
                           shuffle=True, batch_size=1, collate_fn=collate)
    test_set = DataLoader(TMFDatasetAdapter(dataset['Q1 2023':'Q4 2023'][:5], model.bert_tokenizer))
    val_set = DataLoader(TMFDatasetAdapter(dataset['Q1 2022':'Q4 2022'][:5], model.bert_tokenizer))
    logging.info("Loaded datasets (train: %d %2.2f, val: %d %2.2f, test: %d %2.2f)",
                 len(train_set), len(train_set) / len(dataset) * 100,
                 len(val_set), len(val_set) / len(dataset) * 100,
                 len(test_set), len(test_set) / len(dataset) * 100)

    loss_fn = F.cross_entropy
    optimizer = Adagrad(model.parameters(), lr=1e-3)

    logging.info('Start training for %d epochs', epochs)
    for epoch in trange(epochs, desc="Epoch", leave=False):
        train(model, train_set, loss_fn, optimizer, device)
        test(model, val_set, loss_fn, device)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=MotleyFoolDataset, default='./TMF_dataset_annotated.zip')
    parser.add_argument('--weights', type=str, default='ProsusAI/finbert')
    parser.add_argument('--epochs', type=int, default='10')
    args = parser.parse_args()

    main(args.dataset, args.weights, args.epochs)


