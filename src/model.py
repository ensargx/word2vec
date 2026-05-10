import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.u_embeddings = nn.Embedding(vocab_size, emb_dim) # Center words
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim) # Context & Negative words

        init_range = 0.5 / emb_dim
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.zero_()

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)     # [batch, dim]
        emb_v = self.v_embeddings(pos_v)     # [batch, dim]
        emb_neg = self.v_embeddings(neg_v)   # [batch, k, dim]

        # Pozitif Skor: dot(u, v)
        pos_score = torch.sum(torch.mul(emb_u, emb_v), dim=1) 
        pos_score = torch.clamp(pos_score, max=10, min=-10) # Sayısal stabilite
        pos_loss = F.logsigmoid(pos_score)

        # Negatif Skor: u * neg_v^T
        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        return -(pos_loss + neg_loss).mean()
