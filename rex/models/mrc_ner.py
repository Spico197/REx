from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel

from rex.utils.position import extract_positions_from_start_end_label


class PlmMRCModel(nn.Module):
    def __init__(self, plm_dir: str, dropout: Optional[float] = 0.5):
        super().__init__()

        self.plm = BertModel.from_pretrained(plm_dir)
        hidden_size = self.plm.config.hidden_size

        self.start_dense = nn.Linear(hidden_size, 2, bias=True)
        self.end_dense = nn.Linear(hidden_size, 2, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def forward(
        self, input_ids, mask, start_labels=None, end_labels=None, decode=True, **kwargs
    ):
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.dropout(hidden)
        start_logits = self.start_dense(hidden)
        end_logits = self.end_dense(hidden)

        results = {"start_logits": start_logits, "end_logits": end_logits}
        if start_labels is not None:
            start_loss = self.criterion(
                start_logits.reshape(-1, 2), start_labels.reshape(-1)
            )
            end_loss = self.criterion(end_logits.reshape(-1, 2), end_labels.reshape(-1))
            loss = start_loss + end_loss
            results["loss"] = loss

        if decode:
            batch_positions = self.decode(start_logits, end_logits, mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(self, start_logits, end_logits, mask, **kwargs):
        dtype = start_logits.dtype
        slogits = start_logits.detach()
        slogits[..., 1] = slogits[..., 1].masked_fill(
            mask.ne(3), torch.finfo(dtype).min
        )
        elogits = end_logits.detach()
        elogits[..., 1] = elogits[..., 1].masked_fill(
            mask.ne(3), torch.finfo(dtype).min
        )

        start_pred = slogits.max(dim=-1)[1]
        end_pred = elogits.max(dim=-1)[1]

        batch_preds = []
        for start_idxes, end_idxes, raw_tokens, ent_type, ent_offset in zip(
            start_pred,
            end_pred,
            kwargs["raw_tokens"],
            kwargs["ent_type"],
            kwargs["ent_offset"],
        ):
            pred_ents = []
            positions = extract_positions_from_start_end_label(start_idxes, end_idxes)
            for s, e in positions:
                new_s = s - ent_offset
                new_e = e - ent_offset + 1
                pred_ents.append(
                    ("".join(raw_tokens[new_s:new_e]), ent_type, (new_s, new_e))
                )
            batch_preds.append(pred_ents)

        return batch_preds
