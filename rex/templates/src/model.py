import torch
import torch.nn as nn
import torch.nn.functional as F

from rex.modules.crf import PlainCRF
from rex.modules.embeddings.static_embedding import StaticEmbedding


class LSTMCRFModel(nn.Module):
    def __init__(
        self, vocab_size, emb_size, hidden_size, num_lstm_layers, num_tags, dropout
    ):
        super().__init__()

        self.token_emb = StaticEmbedding(vocab_size, emb_size, dropout=dropout)
        self.lstm_enc = nn.LSTM(
            emb_size,
            hidden_size // 2,
            num_layers=num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.hidden2tag = nn.Linear(hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)
        self.num_tags = num_tags
        # self.crf = PlainCRF(num_tags)

    def forward(self, token_ids, mask=None, labels=None, **kwargs):
        emb = self.token_emb(token_ids)
        # out, (_, _) = self.lstm_enc(emb)
        # out = self.dropout(out)
        out = self.hidden2tag(emb)

        # results = {"pred": self.crf.decode(out)}
        results = {"pred": out.argmax(-1).detach().cpu().tolist()}
        if labels is not None:
            labels[mask == 0] = -1
            results["loss"] = F.cross_entropy(
                out.reshape(-1, self.num_tags), labels.reshape(-1), ignore_index=-1
            )
            # -self.crf(out, labels, mask, reduction="mean")
            # results["loss"] = -self.crf(out, labels, mask, reduction="mean")

        return results
