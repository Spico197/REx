from typing import Optional

import torch
import torch.nn as nn

from rex.utils.io import load_embedding_file


class StaticEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout: Optional[float] = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, token_ids):
        rep = self.embedding(token_ids)
        rep = self.dropout(rep)
        return rep

    @classmethod
    def from_pretrained(
        cls,
        filepath: str,
        file_encoding: Optional[str] = "utf8",
        freeze: Optional[bool] = False,
    ):
        tokens, token2emb = load_embedding_file(filepath, encoding=file_encoding)
        weights = []
        for token in tokens:
            weights.append(token2emb[token])
        weights = torch.tensor(weights, dtype=torch.float)
        embed = cls(weights.shape[0], weights.shape[1])
        embed.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
        return embed
