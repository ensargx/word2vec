from collections import Counter
import argparse
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

from src.config import *
from src.utils import set_seed, split_data, subsampling, create_unigram_table
from src.dataset import SkipGramDataset, load_processed_data, save_processed_data
from src.model import SkipGramModel
from src.trainer import Trainer

def load_or_process_data():
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
    return train_ready, word2idx, unigram_table, vocab_size

def training():
    set_seed(42)

    train_ready, word2idx, unigram_table, vocab_size = load_or_process_data()
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

def get_embeddings_and_vocabs(dim, epoch):
    """Sadece gerekli olan embedding tensörünü ve sözlükleri yükler."""
    model_path = os.path.join(CHECKPOINT_DIR, f"dim_{dim}", f"model_e{epoch}.pt")

    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    key = "u_embeddings.weight" if "u_embeddings.weight" in state_dict else "_orig_mod.u_embeddings.weight"
    return state_dict[key]

def evaluate_analogy_rank(embeddings, word2idx, idx2word, a, b, c, expected):
    """
    v(a) - v(b) + v(c) işlemini yapar, girdi kelimelerini (a, b, c) sonuçtan dışlar.
    Girdi:
        embeddings: Ham embedding tensörü [vocab_size, emb_dim]
        word2idx, idx2word: Sözlükler
        a, b, c, expected: Analoji kelimeleri
    Çıktı:
        rank: Beklenen kelimenin filtrelenmiş listedeki sırası
        top_word: Filtrelenmiş listedeki 1. sıradaki kelime
        similarity: Beklenen kelimenin kosinüs benzerlik skoru (0-1 arası)
    """
    input_words = [a.lower(), b.lower(), c.lower()]
    target_word = expected.lower()

    all_words = input_words + [target_word]
    if any(w not in word2idx for w in all_words):
        return None, "missing", 0.0

    idx_a, idx_b, idx_c, idx_exp = [word2idx[w] for w in all_words]
    target_vec = (embeddings[idx_a] - embeddings[idx_b] + embeddings[idx_c]).unsqueeze(0)

    sims = F.cosine_similarity(embeddings, target_vec)

    raw_sim = sims[idx_exp].item()
    norm_sim = (raw_sim + 1) / 2

    eval_sims = sims.clone()
    for idx in [idx_a, idx_b, idx_c]:
        eval_sims[idx] = -float('inf')

    sorted_indices = torch.argsort(eval_sims, descending=True)
    rank = (sorted_indices == idx_exp).nonzero(as_tuple=True)[0].item() + 1
    top_word = idx2word[sorted_indices[0].item()]

    return rank, top_word, norm_sim

def results():
    dims = [128, 256, 512]
    epochs = [10, 15]
    analogies_dict = {
        "Aile ve Cinsiyet": [
            ("king", "man", "woman", "queen"),
            ("prince", "boy", "girl", "princess"),
            ("uncle", "man", "woman", "aunt"),
            ("son", "boy", "girl", "daughter"),
            ("brother", "man", "woman", "sister")
        ],
        "Başkentler": [
            ("turkey", "ankara", "france", "paris"),
            ("germany", "berlin", "italy", "rome"),
            ("russia", "moscow", "spain", "madrid"),
            ("japan", "tokyo", "egypt", "cairo"),
            ("greece", "athens", "china", "beijing")
        ],
        "Sıfatlar": [
            ("bigger", "big", "smaller", "small"),
            ("faster", "fast", "slower", "slow"),
            ("better", "good", "worse", "bad"),
            ("stronger", "strong", "weaker", "weak"),
            ("harder", "hard", "easier", "easy")
        ],
        "Fiil Çekimleri": [
            ("ate", "eat", "played", "play"),
            ("going", "go", "walking", "walk"),
            ("took", "take", "gave", "give"),
            ("swimming", "swim", "running", "run"),
            ("sang", "sing", "danced", "dance")
        ]
    }
    _, word2idx, _, _ = load_or_process_data()
    idx2word = {idx: word for word, idx in word2idx.items()}
    all_final_results = []

    for dim in dims:
        for epoch in epochs:
            embeddings = get_embeddings_and_vocabs(dim, epoch-1)
            for cat, pairs in analogies_dict.items():
                for a, b, c, expected in pairs:
                    rank, top_word, norm_sim = evaluate_analogy_rank(embeddings, word2idx, idx2word, a, b, c, expected)
                    all_final_results.append({
                        "Boyut": dim,
                        "Epoch": epoch,
                        "Kategori": cat,
                        "Soru (Analoji)": f"{a}-{b}+{c}",
                        "Beklenen Kelime": expected,
                        "Modelin Tahmini": top_word if rank else "Eksik Kelime",
                        "Sıralama (Rank)": rank if rank else "N/A",
                        "Benzerlik": norm_sim
                    })
                    print(f'dim: {dim} | epoch: {epoch} | analogy: {a}-{b}+{c} | expected: {expected} | rank: {rank} | find: {top_word}')

    df = pd.DataFrame(all_final_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False, encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser(
        description="Word2Vec Research Implementation"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("training", help="Run training pipeline")
    subparsers.add_parser("results", help="Run results evaluation")
    args = parser.parse_args()
    if args.mode == "training":
        training()
    elif args.mode == "results":
        results()

if __name__ == "__main__":
    main()
