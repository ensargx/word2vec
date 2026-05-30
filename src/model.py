import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, unigram_probs, k_neg=5):
        super().__init__()
        self.k_neg = k_neg

        self.register_buffer("unigram_probs", torch.tensor(unigram_probs, dtype=torch.float32))

        self.u_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim)

        initrange = 1.0 / emb_dim
        nn.init.uniform_(self.u_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.v_embeddings.weight)

    def forward(self, pos_u, pos_v):
        batch_size = pos_u.size(0)

        neg_ids = torch.multinomial(
            self.unigram_probs,
            num_samples=batch_size * self.k_neg,
            replacement=True
        ).view(batch_size, self.k_neg)

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg = self.v_embeddings(neg_ids)

        pos_score = torch.sum(emb_u * emb_v, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        return -(pos_loss + neg_loss).mean()