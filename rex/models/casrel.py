from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert import BertModel

from rex.modules.span import SubjObjSpan
from rex.modules.embeddings.static_embedding import StaticEmbedding


class LSTMCasRel(nn.Module):
    def __init__(
        self,
        vocab,
        num_lstm_layers,
        dim_token_emb,
        num_classes,
        pred_threshold: Optional[float] = 0.5,
        emb_filepath: Optional[str] = None,
        dropout: Optional[float] = 0.5,
    ):
        super().__init__()
        if dim_token_emb % 2 != 0:
            raise ValueError("Dimension of ``dim_token_emb`` must be odd")

        self.token_embedding = StaticEmbedding(
            vocab, dim_token_emb, filepath=emb_filepath, dropout=dropout
        )
        self.encoder = nn.LSTM(
            dim_token_emb,
            dim_token_emb // 2,
            num_lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.span_nn = SubjObjSpan(dim_token_emb, num_classes, pred_threshold)

    def encode(self, hidden, mask):
        batch_size, seq_len, hidden_size = hidden.shape
        total_length = mask.shape[1]
        lens = mask.sum(dim=1)

        x = pack_padded_sequence(
            hidden, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (_, _) = self.encoder(x)  # h_n: #(direction, #sent, hidden_size)
        output, _ = pad_packed_sequence(
            output, batch_first=True, padding_value=0.0, total_length=total_length
        )
        return output

    def forward(
        self,
        token_ids,
        mask,
        subj_heads=None,
        subj_tails=None,
        subj_head=None,
        subj_tail=None,
        obj_head=None,
        obj_tail=None,
    ):
        result = {}
        emb = self.token_embedding(token_ids)
        hidden = self.encode(emb, mask)
        # hidden = emb

        if self.training:
            (
                subj_head_logits,
                subj_tail_logits,
                obj_head_logits,
                obj_tail_logits,
            ) = self.span_nn(hidden, subj_head, subj_tail)

            subj_head_loss = F.binary_cross_entropy_with_logits(
                subj_head_logits, subj_heads, reduction="none"
            )
            subj_head_loss = (subj_head_loss * mask).sum() / mask.sum()
            subj_tail_loss = F.binary_cross_entropy_with_logits(
                subj_tail_logits, subj_tails, reduction="none"
            )
            subj_tail_loss = (subj_tail_loss * mask).sum() / mask.sum()
            subj_loss = subj_head_loss + subj_tail_loss

            result["subj_loss"] = subj_loss
            result["subj_head_loss"] = subj_head_loss
            result["subj_tail_loss"] = subj_tail_loss

            obj_head_loss = F.binary_cross_entropy_with_logits(
                obj_head_logits, obj_head, reduction="none"
            )
            obj_head_loss = (obj_head_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            obj_tail_loss = F.binary_cross_entropy_with_logits(
                obj_tail_logits, obj_tail, reduction="none"
            )
            obj_tail_loss = (obj_tail_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            obj_loss = obj_head_loss + obj_tail_loss

            result["obj_loss"] = obj_loss
            result["obj_head_loss"] = obj_head_loss
            result["obj_tail_loss"] = obj_tail_loss

            result["loss"] = subj_loss + obj_loss
        else:
            triples = self.span_nn.predict(hidden)
            result["pred"] = triples

        return result


class CasRel(nn.Module):
    def __init__(
        self,
        bert_model_dir,
        num_classes,
        pred_threshold: Optional[float] = 0.5,
    ):
        super().__init__()
        self.pred_threshold = pred_threshold

        self.bert_encoder = BertModel.from_pretrained(bert_model_dir)
        self.hidden_size = self.bert_encoder.config.hidden_size
        self.span_nn = SubjObjSpan(self.hidden_size, num_classes, pred_threshold)

    def encode(self, input_ids, attention_mask):
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(
        self,
        token_ids,
        mask,
        subj_heads=None,
        subj_tails=None,
        subj_head=None,
        subj_tail=None,
        obj_head=None,
        obj_tail=None,
    ):
        result = {}
        hidden = self.encode(token_ids, mask)

        if self.training:
            (
                subj_head_logits,
                subj_tail_logits,
                obj_head_logits,
                obj_tail_logits,
            ) = self.span_nn(hidden, subj_head, subj_tail)

            subj_head_loss = F.binary_cross_entropy_with_logits(
                subj_head_logits, subj_heads, reduction="none"
            )
            subj_head_loss = (subj_head_loss * mask).sum() / mask.sum()
            subj_tail_loss = F.binary_cross_entropy_with_logits(
                subj_tail_logits, subj_tails, reduction="none"
            )
            subj_tail_loss = (subj_tail_loss * mask).sum() / mask.sum()
            subj_loss = subj_head_loss + subj_tail_loss

            result["subj_loss"] = subj_loss
            result["subj_head_loss"] = subj_head_loss
            result["subj_tail_loss"] = subj_tail_loss

            obj_head_loss = F.binary_cross_entropy_with_logits(
                obj_head_logits, obj_head, reduction="none"
            )
            obj_head_loss = (obj_head_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            obj_tail_loss = F.binary_cross_entropy_with_logits(
                obj_tail_logits, obj_tail, reduction="none"
            )
            obj_tail_loss = (obj_tail_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            obj_loss = obj_head_loss + obj_tail_loss

            result["obj_loss"] = obj_loss
            result["obj_head_loss"] = obj_head_loss
            result["obj_tail_loss"] = obj_tail_loss

            result["loss"] = subj_loss + obj_loss
        else:
            triples = self.span_nn.predict(hidden)
            result["pred"] = triples

        return result
