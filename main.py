import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.config import cfg
from src.dataset import SkipGramDataset, load_or_process_data
from src.model import SkipGramModel
from src.trainer import Trainer

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def get_embeddings_and_vocabs(dim, epoch):
    """Sadece gerekli olan embedding tensörünü ve sözlükleri yükler."""
    model_path = os.path.join(cfg.paths.checkpoint_dir, f"dim_{dim}", f"model_e{epoch}.pt")

    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
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
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.paths.results_dir, "results.csv"), index=False, encoding='utf-8-sig')

def plot_results():
    dims = [128, 256, 512]
    os.makedirs(cfg.paths.results_dir, exist_ok=True)

    f_comp, a_comp = plt.subplots(figsize=(12, 7))

    for dim in dims:
        p = os.path.join(cfg.paths.checkpoint_dir, f"dim_{dim}", "train_log.csv")
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        mb = df['batch'].max() + 1
        df['s'] = df['epoch'] * mb + df['batch']
        df['m'] = df['loss'].rolling(window=200, min_periods=1).mean()

        final_loss = df['m'].iloc[-1]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(df['s'], df['loss'], alpha=0.45, color='steelblue', lw=1.0)
        ax.plot(df['s'], df['m'], color='midnightblue', lw=2.5, label=f'Dim {dim} Trend')

        ax.axhline(y=final_loss, color='red', linestyle='--', lw=1.5, alpha=0.8, 
                   label=f'Final Loss: {final_loss:.4f}')

        ax.set_title(f'Loss Performance (Dim {dim})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, ls=':', alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.paths.results_dir, f"loss_dim{dim}.png"), dpi=300)
        plt.close(fig)

        a_comp.plot(df['s'], df['m'], label=f'Dimension {dim}', lw=2.2)

    a_comp.set_title('Cross-Dimension Loss Comparison', fontsize=15, fontweight='bold')
    a_comp.set_xlabel('Steps')
    a_comp.set_ylabel('Smoothed Loss')
    a_comp.grid(True, ls=':', alpha=0.6)
    a_comp.legend()
    f_comp.tight_layout()
    f_comp.savefig(os.path.join(cfg.paths.results_dir, "loss_comparison.png"), dpi=300)
    plt.close(f_comp)

def main():
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram Research Implementation")
    parser.add_argument("command", choices=["train", "results", "plot"], help="Command to execute.")

    args = parser.parse_args()
    if args.command == "train":
        training()
    elif args.command == "results":
        results()
    elif args.command == "plot":
        plot_results()


if __name__ == "__main__":
    main()
