from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from rex.data.vocab import Vocab
from rex.utils.io import load_embedding_file


class StaticEmbedding(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        dim_token_emb: int,
        filepath: Optional[str] = None,
        dropout: Optional[float] = 0.3,
        freeze: Optional[bool] = False
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        if filepath is None:
            # random initialized
            self.embedding = nn.Embedding(vocab.size, dim_token_emb)
            logger.info(f"Random initialized embeddings, vocab size: {vocab.size}")
        else:
            token2emb = load_embedding_file(filepath)
            weights = []
            in_emb_file = 0
            for token_id in range(vocab.size):
                token = vocab.id2token[token_id]
                if token in token2emb:
                    weights.append(token2emb[token])
                    in_emb_file += 1
                else:
                    weights.append(np.random.randn(dim_token_emb).tolist())
            weights = torch.tensor(weights, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(
                weights, freeze=freeze, padding_idx=vocab.pad_idx)
            logger.info(("Embedding initialized from emb file, "
                         f"vocab size: {vocab.size}, "
                         f"emb file vocab size: {len(token2emb)}, "
                         f"{vocab.size - in_emb_file} tokens are not in the "
                         "emb file and are randomly initialized."))

    def forward(self, token_ids):
        rep = self.embedding(token_ids)
        rep = self.dropout(rep)
        return rep
