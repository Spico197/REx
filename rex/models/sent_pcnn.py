from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rex.modules.pcnn import PiecewiseCNN
from rex.modules.embeddings.static_embedding import StaticEmbedding


class SentPCNN(nn.Module):
    def __init__(
        self,
        vocab,
        emb_filepath,
        num_classes,
        dim_token_emb,
        pos_emb_capacity,
        dim_pos,
        num_filters,
        kernel_size,
        dropout: Optional[float] = 0.5,
    ):
        super().__init__()

        self.token_embedding = StaticEmbedding(
            vocab, dim_token_emb, emb_filepath, dropout=dropout
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=pos_emb_capacity, embedding_dim=dim_pos
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=pos_emb_capacity, embedding_dim=dim_pos
        )
        self.pcnn = PiecewiseCNN(
            dim_token_emb + 2 * dim_pos, num_filters, kernel_size, dropout=dropout
        )
        self.dense = nn.Linear(
            in_features=num_filters * 3, out_features=num_classes, bias=True
        )

    def forward(self, token_ids, head_pos, tail_pos, mask, labels=None):
        token_embedding = self.token_embedding(token_ids)
        pos1_embedding = self.pos1_embedding(head_pos)
        pos2_embedding = self.pos2_embedding(tail_pos)
        x = torch.cat([token_embedding, pos1_embedding, pos2_embedding], dim=-1)
        x = self.pcnn(x, mask)
        out = self.dense(x)

        result = {"pred": torch.sigmoid(out)}
        if labels is not None:
            result.update(
                {"loss": F.binary_cross_entropy_with_logits(out, labels.float())}
            )

        return result
