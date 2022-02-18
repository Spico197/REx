import torch
import torch.nn as nn
from rex.modules.crf import PlainCRF


class LSTMCRFModel(nn.Module):
    def __init__(
        self, vocab_size, emb_size, hidden_size, num_lstm_layers, num_tags, dropout
    ):
        super().__init__()

        self.emb_enc = nn.Embedding(vocab_size, emb_size)
        self.lstm_enc = nn.LSTM(
            emb_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.hidden2tag = nn.Linear(2 * hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)
        self.crf = PlainCRF(num_tags)

    def forward(self, input_ids, labels=None, mask=None, **kwargs):
        emb = self.emb_enc(input_ids)
        emb = self.dropout(emb)
        sorted_seq_lengths, indices = torch.sort(mask.sum(-1), descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        emb = emb[indices]
        out = nn.utils.rnn.pack_padded_sequence(
            emb, sorted_seq_lengths.detach().cpu(), batch_first=True
        )
        out, (_, _) = self.lstm_enc(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=mask.size(-1)
        )
        out = out[desorted_indices]
        out = self.hidden2tag(out)

        results = {"preds": self.crf.decode(out)}
        if mask is not None:
            mask = mask.bool()
        if labels is not None:
            results["loss"] = -self.crf(out, labels, mask)

        return results
