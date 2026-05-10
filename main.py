from collections import Counter
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from src.config import *
from src.utils import split_data, subsampling, create_unigram_table
from src.dataset import SkipGramDataset
from src.model import SkipGramModel
from src.trainer import Trainer

def main():
    ds = load_dataset(DATASET_NAME)
    train_raw, _, _ = split_data(ds)
    full_counts = Counter(train_raw)
    train_ready = subsampling([w for w in train_raw if full_counts[w] >= 5])

    vocab = sorted(list(set(train_ready)))
    word2idx = {word: i for i, word in enumerate(vocab)}
    unigram_table = create_unigram_table({w: full_counts[w] for w in vocab}, vocab)

    model = SkipGramModel(len(vocab), EMB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, optimizer, log_interval=100)
    start_epoch = trainer.load_latest_checkpoint()

    dataset = SkipGramDataset(train_ready, word2idx, unigram_table, k=K_NEG, window_size=5)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    trainer.train(loader, start_epoch=start_epoch)

if __name__ == "__main__":
    main()
