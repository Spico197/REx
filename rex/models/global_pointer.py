import torch
import torch.nn as nn
from transformers.models.bert import BertModel

from rex.modules.affine import Biaffine
from rex.modules.ffn import FFN
from rex.utils.position import decode_multi_class_pointer_mat_span


class PointerMatrix(nn.Module):
    """Pointer Matrix Prediction

    References:
        - https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/models/GlobalPointer.py
    """

    def __init__(
        self,
        hidden_size,
        biaffine_size,
        cls_num=2,
        dropout=0,
        biaffine_bias=False,
        use_rope=False,
    ):
        super().__init__()
        self.linear_h = FFN(hidden_size, biaffine_size, dropout=dropout)
        self.linear_t = FFN(hidden_size, biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(
            n_in=biaffine_size,
            n_out=cls_num,
            bias_x=biaffine_bias,
            bias_y=biaffine_bias,
        )
        self.use_rope = use_rope

    def sinusoidal_position_embedding(self, qw, kw):
        batch_size, seq_len, output_dim = qw.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pos_emb = position_ids * indices
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = pos_emb.repeat((batch_size, *([1] * len(pos_emb.shape))))
        pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, output_dim))
        pos_emb = pos_emb.to(qw)

        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], -1)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], -1)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, x):
        h = self.linear_h(x)
        t = self.linear_t(x)
        if self.use_rope:
            h, t = self.sinusoidal_position_embedding(h, t)
        o = self.biaffine(h, t)
        return o


def multilabel_categorical_crossentropy(y_pred, y_true, reduction="mean"):
    """
    https://kexue.fm/archives/7359
    https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    if reduction == "mean":
        return (neg_loss + pos_loss).mean()
    elif reduction == "sum":
        return (neg_loss + pos_loss).sum()
    else:
        return neg_loss + pos_loss


class GlobalPointer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        cls_num: int,
        hidden_size: int,
        biaffine_size: int = 512,
        use_rope: bool = True,
        dropout: float = 0.3,
        tri_mask: str = "tril",
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        self.hidden_size = hidden_size
        self.biaffine_size = biaffine_size
        self.use_rope = use_rope
        self.tri_mask = tri_mask
        assert tri_mask in ["tril", "triu", "none"]

        self.encoder = encoder
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=cls_num,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        bs, seq_len = input_ids.shape
        attention_mask = mask.gt(0).float()
        pad_mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(bs, self.cls_num, seq_len, seq_len)
        )

        hidden = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.pointer(hidden)

        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        if self.tri_mask == "tril":
            tri_mask = torch.tril(torch.ones_like(logits), -1)
        elif self.tril_mask == "triu":
            tri_mask = torch.triu(torch.ones_like(logits), -1)
        else:
            tri_mask = torch.zeros_like(logits)
        logits = logits - tri_mask * 1e12
        logits = logits / self.biaffine_size**0.5

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * self.cls_num, -1),
                labels.reshape(bs * self.cls_num, -1),
            )
            results["loss"] = loss
        if is_eval:
            batch_positions = self.decode(logits, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(
        self,
        logits: torch.Tensor,
        **kwargs,
    ):
        pred = (logits > 0).long()
        batch_preds = decode_multi_class_pointer_mat_span(
            pred, offsets=kwargs.get("offset")
        )
        return batch_preds


class BertTokenEncoder(nn.Module):
    def __init__(self, plm_dir: str) -> None:
        super().__init__()

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outs = self.plm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )
        hidden = outs.last_hidden_state
        return hidden


class BertGlobalPointer(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int,
        biaffine_size: int = 512,
        use_rope=True,
        dropout=0.0,
        tri_mask: str = "tril",
    ) -> None:
        super().__init__()

        encoder = BertTokenEncoder(plm_dir)
        self.global_pointer = GlobalPointer(
            encoder,
            encoder.hidden_size,
            cls_num,
            biaffine_size=biaffine_size,
            use_rope=use_rope,
            dropout=dropout,
            tri_mask=tri_mask,
        )

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        results = self.global_pointer(
            input_ids, mask, labels=labels, is_eval=is_eval, **kwargs
        )
        return results
