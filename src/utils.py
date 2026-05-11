import torch
import os
import numpy as np
from collections import Counter
import yaml

from src.config import CHECKPOINT_DIR

def split_data(ds):
    train, test, validation = ds["train"], ds["test"], ds["validation"]
    split_text = lambda dataset: [x for x in dataset["text"][0].split() if x]
    return split_text(train), split_text(test), split_text(validation)

def subsampling(words, threshold=1e-5):
    counts = Counter(words)
    total_count = len(words)
    new_words = []
    for w in words:
        freq = counts[w] / total_count
        p_keep = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
        if np.random.random() < p_keep:
            new_words.append(w)
    return new_words

def create_unigram_table(word_counts, vocab, table_size=100_000_000):
    freqs = np.array([word_counts[w] for w in vocab])
    pow_freqs = freqs**0.75
    probs = pow_freqs / sum(pow_freqs)

    table = np.zeros(table_size, dtype=np.int32)
    count = 0
    for idx, p in enumerate(probs):
        n = int(p * table_size)
        table[count : count + n] = idx
        count += n
    return table

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def list_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    return sorted(files, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
