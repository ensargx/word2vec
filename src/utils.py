import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def subsampling(corpus_ids, counts, threshold=1e-5):
    counts = np.asarray(counts, dtype=np.float64)
    freqs = counts / counts.sum()

    p_keep = np.ones_like(freqs)
    valid = freqs > 0
    p_keep[valid] = (np.sqrt(freqs[valid] / threshold) + 1.0) * (threshold / freqs[valid])
    p_keep = np.minimum(p_keep, 1.0)

    subsampled = []

    for sentence in corpus_ids:
        sent = np.asarray(sentence, dtype=np.int64)
        keep_mask = np.random.random(sent.shape[0]) < p_keep[sent]
        new_sent = sent[keep_mask]

        if len(new_sent) > 1:
            subsampled.append(new_sent.tolist())

    return subsampled


def build_unigram_probs(counts, power=0.75):
    probs = counts.astype(np.float64) ** power
    probs /= probs.sum()
    return probs