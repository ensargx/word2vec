import os
import joblib
import numpy as np
import torch
import torch.optim as optim

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import cfg
from src.tokenizer import Tokenizer
from src.dataset import SkipGramDataset
from src.model import SkipGramModel
from src.trainer import Trainer
from src.utils import set_seed, ensure_parent_dir, subsampling, build_unigram_probs


def artifacts_exist():
    return os.path.exists(cfg.paths.subsampled) and os.path.exists(cfg.paths.unigram_probs)


def build_artifacts():
    ds = load_dataset("parquet", data_files=cfg.paths.dataset, split="train")

    tokenizer = Tokenizer(min_count=cfg.model.min_count)
    tokens, counter = tokenizer.fit(ds)

    encoded = tokenizer.encode(tokens)
    counts = np.fromiter((counter.get(w, 0) for w in tokenizer.idx2word), dtype=np.int64)

    subsampled = subsampling(encoded, counts, threshold=cfg.training.subsampling_threshold)
    unigram_probs = build_unigram_probs(counts)

    ensure_parent_dir(cfg.paths.tokenizer)
    ensure_parent_dir(cfg.paths.subsampled)
    ensure_parent_dir(cfg.paths.unigram_probs)

    tokenizer.save(cfg.paths.tokenizer)
    joblib.dump(subsampled, cfg.paths.subsampled, compress=3)
    joblib.dump(unigram_probs, cfg.paths.unigram_probs, compress=3)

    return subsampled, unigram_probs


def load_or_build_artifacts():
    if artifacts_exist():
        subsampled = joblib.load(cfg.paths.subsampled)
        unigram_probs = joblib.load(cfg.paths.unigram_probs)
        return subsampled, unigram_probs

    return build_artifacts()


def build_loader(subsampled):
    dataset = SkipGramDataset(subsampled, window_size=cfg.training.window_size)
    num_workers = cfg.num_workers

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )


def build_model(unigram_probs):
    model = SkipGramModel(
        vocab_size=cfg.model.vocab_size,
        emb_dim=cfg.model.emb_dim,
        unigram_probs=unigram_probs,
        k_neg=cfg.model.k_neg,
        sparse=cfg.model.sparse,
    ).to(cfg.device)

    if cfg.training.compile:
        model = torch.compile(model, mode=cfg.training.compile_mode)

    return model


def build_optimizer(model):
    optim_cfg = vars(cfg.training.optim).copy()
    optim_name = optim_cfg.pop("name")

    if not hasattr(optim, optim_name):
        raise ValueError(f"Unknown optimizer: {optim_name}")

    return getattr(optim, optim_name)(model.parameters(), **optim_cfg)


def main():
    set_seed(cfg.seed)

    subsampled, unigram_probs = load_or_build_artifacts()

    loader = build_loader(subsampled)
    model = build_model(unigram_probs)
    optimizer = build_optimizer(model)

    trainer = Trainer(model=model, optimizer=optimizer, cfg=cfg)
    trainer.fit(loader)


if __name__ == "__main__":
    main()
