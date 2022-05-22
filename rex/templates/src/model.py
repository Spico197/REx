import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from rex.modules.crf import PlainCRF


class LSTMCRFModel(nn.Module):
    def __init__(self, plm_dir, num_lstm_layers, num_tags, dropout):
        super().__init__()

        self.num_tags = num_tags

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size
        # self.lstm_enc = nn.LSTM(
        #     self.hidden_size,
        #     self.hidden_size // 2,
        #     num_layers=num_lstm_layers,
        #     bias=True,
        #     batch_first=True,
        #     dropout=dropout,
        #     bidirectional=True,
        # )
        self.hidden2tag = nn.Linear(self.hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)
        # self.crf = PlainCRF(num_tags)

    def forward(self, token_ids, mask=None, labels=None, **kwargs):
        out = self.plm(input_ids=token_ids, attention_mask=mask, return_dict=True)
        out = out.last_hidden_state
        # out, (_, _) = self.lstm_enc(emb)
        # out = self.dropout(out)
        out = self.hidden2tag(out)

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
