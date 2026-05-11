import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, unigram_table, k_neg=5):
        super().__init__()
        self.k_neg = k_neg

        self.register_buffer('unigram_table', torch.LongTensor(unigram_table))
        self.table_size = len(unigram_table)

        self.u_embeddings = nn.Embedding(vocab_size, emb_dim) # Center words
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim) # Context words

        initrange = 1.0 / emb_dim
        nn.init.uniform_(self.u_embeddings.weight, -initrange, initrange)
        nn.init.uniform_(self.v_embeddings.weight, -0, 0)

    def forward(self, pos_u, pos_v):
        batch_size = pos_u.size(0)

        random_indices = torch.randint(0, self.table_size, (batch_size, self.k_neg), device=pos_u.device)
        neg_v = self.unigram_table[random_indices]
        neg_v = neg_v.view(batch_size, self.k_neg)

        emb_u = self.u_embeddings(pos_u)     # [B, dim]
        emb_v = self.v_embeddings(pos_v)     # [B, dim]
        neg_v = self.v_embeddings(neg_v) # [B, K, dim]

        pos_score = torch.sum(emb_u * emb_v, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_v, emb_u.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        return -(pos_loss + neg_loss).mean()
