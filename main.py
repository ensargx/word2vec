import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datasets import load_dataset

from src.config import cfg
from src.dataset import SkipGramDataset, load_or_process_data
from src.model import SkipGramModel
from src.trainer import Trainer
from src.tokenizer import Tokenizer
from src.utils import subsampling

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_or_process_data():
    ...

def training():
    set_seed(42)

    train_ready, word2idx, unigram_table, vocab_size = load_or_process_data()
    model = SkipGramModel(
        vocab_size=vocab_size,
        emb_dim=cfg.training.emb_dim,
        unigram_table=unigram_table,
        k_neg=cfg.training.k_neg
    ).to(cfg.device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    trainer = Trainer(model, optimizer, cfg.training.emb_dim, log_interval=100)
    start_epoch = trainer.load_latest_checkpoint()

    train_indices = [word2idx[w] for w in train_ready]
    dataset = SkipGramDataset(train_indices, window_size=5)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True, 
        shuffle=False
    )

    trainer.train(loader, start_epoch=start_epoch, max_epochs=cfg.training.epochs)

def main():
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram Research Implementation")
    parser.add_argument("command", choices=["train"], help="Command to execute.")

    args = parser.parse_args()
    if args.command == "train":
        training()

if __name__ == "__main__":
    main()
