from collections import Counter
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from src.config import *
from src.utils import set_seed, split_data, subsampling, create_unigram_table
from src.dataset import SkipGramDataset, load_processed_data, save_processed_data
from src.model import SkipGramModel
from src.trainer import Trainer

def main():
    set_seed(42)
    bundle = load_processed_data()

    if bundle:
        train_ready = bundle['train_ready']
        word2idx = bundle['word2idx']
        unigram_table = bundle['unigram_table']
        vocab_size = bundle['vocab_size']
    else:
        ds = load_dataset(DATASET_NAME)
        train_raw, test_raw, validation_raw = split_data(ds)
        train_raw = train_raw + test_raw + validation_raw
        full_counts = Counter(train_raw)

        vocab = sorted([w for w, count in full_counts.items() if count >= 5])
        vocab_size = len(vocab)
        word2idx = {word: i for i, word in enumerate(vocab)}

        unigram_table = create_unigram_table({w: full_counts[w] for w in vocab}, vocab)
        train_ready = subsampling([w for w in train_raw if full_counts[w] >= 5])

        bundle = {
            'train_ready': train_ready,
            'word2idx': word2idx,
            'unigram_table': unigram_table,
            'vocab_size': vocab_size
        }
        save_processed_data(bundle)

    model = SkipGramModel(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        unigram_table=unigram_table,
        k_neg=K_NEG
    ).to(DEVICE)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(model, optimizer, EMB_DIM, log_interval=100)
    start_epoch = trainer.load_latest_checkpoint()

    train_indices = [word2idx[w] for w in train_ready]
    dataset = SkipGramDataset(train_indices, window_size=5)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True, 
        shuffle=False
    )

    trainer.train(loader, start_epoch=start_epoch, max_epochs=EPOCHS)

if __name__ == "__main__":
    main()
