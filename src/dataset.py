import os
import numpy as np
from torch.utils.data import IterableDataset
import torch

class SkipGramDataset(IterableDataset):
    def __init__(self, data, word2idx, unigram_table, k=5, window_size=5):
        self.data = data
        self.word2idx = word2idx
        self.window_size = window_size
        self.unigram_table = unigram_table
        self.k = k

    def get_negative_samples(self, blacklist):
        extra_size = self.k + len(blacklist) + 2 
        samples = np.random.choice(self.unigram_table, size=extra_size, replace=True)
        valid_samples = [int(s) for s in samples if s not in blacklist]
        while len(valid_samples) < self.k:
            candidate = int(np.random.choice(self.unigram_table))
            if candidate not in blacklist:
                valid_samples.append(candidate)
        return valid_samples[:self.k]

    def __iter__(self):
        for i, word in enumerate(self.data):
            target_idx = self.word2idx[word]
            window = np.random.randint(1, self.window_size + 1)
            start = max(0, i - window)
            end = min(len(self.data), i + window + 1)
            context_positions = [j for j in range(start, end) if j != i]
            if not context_positions: continue
            pairs = [(target_idx, self.word2idx[self.data[context_idx]]) for context_idx in context_positions]

            for pos_u, pos_v in pairs:
                neg_v = self.get_negative_samples([pos_u, pos_v])
                yield torch.tensor(pos_u), torch.tensor(pos_v), torch.tensor(neg_v)

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
    from src.config import DEVICE
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    return data
