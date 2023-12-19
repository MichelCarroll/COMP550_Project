import argparse
import logging

import torch
from torch.utils.data import DataLoader

from src.neural.dataset import TMFDatasetAdapter
from src.dataset_utils import MotleyFoolDataset
from src.neural.model import NeuralNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=MotleyFoolDataset, default='./TMF_dataset_annotated.zip')
    parser.add_argument('--weights', type=str, default='ProsusAI/finbert')
    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights

    logging.info("Loading model with weights from %s", weights)
    model = NeuralNet(fin_bert_weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Running PyTorch on device: %s", device)

    logging.info("Loading datasets")
    train_set = DataLoader(TMFDatasetAdapter(dataset[:'Q4 2021']),
                           shuffle=True, batch_size=400)
    test_set = DataLoader(TMFDatasetAdapter(dataset['Q1 2023':'Q4 2023']))
    val_set = DataLoader(TMFDatasetAdapter(dataset['Q1 2022':'Q4 2022']))
    logging.info("Loaded datasets (train: %d %2.2f, val: %d %2.2f, test: %d %2.2f)",
                 len(train_set), len(train_set) / len(dataset) * 100,
                 len(val_set), len(val_set) / len(dataset) * 100,
                 len(test_set), len(test_set) / len(dataset) * 100)

    for X, y in train_set:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break




