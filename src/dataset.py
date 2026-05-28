from collections import Counter
from src.utils import subsampling, create_unigram_table
from src.config import cfg
from datasets import load_dataset
import os
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import torch

class SkipGramDataset(IterableDataset):
    def __init__(self, data_idxs, window_size=5):
        self.data = data_idxs 
        self.window_size = window_size

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Tek bir worker varsa verinin tamami
            iter_start = 0
            iter_end = len(self.data)
        else:
            # Birden fazla worker varsa veriyi paylastir
            per_worker = int(np.ceil(len(self.data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))

        # Kendi dilimimiz uzerinde donuyoruz
        for i in range(iter_start, iter_end):
            target_idx = self.data[i]

            # Dinamik window size (Word2Vec standarti)
            window = np.random.randint(1, self.window_size + 1)

            start = max(0, i - window)
            end = min(len(self.data), i + window + 1)

            for j in range(start, end):
                if i == j: continue

                # Tensor donusturme isini burada degil, DataLoader'in 
                # default_collate fonksiyonuna birakmak (sadece ham deger dondurerek) daha hizlidir.
                yield target_idx, self.data[j]

def process_data(ds):
    """
    dataset'i al ve process et.
    """
    train = ds["train"]
    return train["text"]

def save_processed_data(data_dict, filename="training_data.pt", data_dir="data"):
    """Sözlük ve eğitim verilerini torch formatında kaydeder."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path = os.path.join(data_dir, filename)
    torch.save(data_dict, path)

def load_processed_data(filename="training_data.pt", data_dir="data"):
    """Torch ile kaydedilmiş paketi geri yükler."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location=cfg.device, weights_only=False)
    return data

def load_or_process_data():
    bundle = load_processed_data()
    if bundle:
        train_ready = bundle['train_ready']
        word2idx = bundle['word2idx']
        unigram_table = bundle['unigram_table']
        vocab_size = bundle['vocab_size']
    else:
        ds = load_dataset(cfg.dataset.path, cfg.dataset.name)
        train_raw = process_data(ds)
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
    return train_ready, word2idx, unigram_table, vocab_size
