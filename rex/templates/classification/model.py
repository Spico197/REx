from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rex.modules.embeddings.static_embedding import StaticEmbedding
from rex.modules.cnn import MultiKernelCNN
from rex.modules.ffn import FFN
from transformers import AutoModel


class DummyModel(nn.Module):
    def __init__(
        self,
        vocab_size: Optional[int] = 5000,
        embedding_dim: Optional[int] = 300,
        in_channel: Optional[int] = 300,
        out_channel: Optional[int] = 300,
        num_classes: Optional[int] = 3,
        dropout: Optional[float] = 0.5,
    ):
        super().__init__()

        self.embedding = StaticEmbedding(vocab_size, embedding_dim, dropout=dropout)
        self.cnn = MultiKernelCNN(
            in_channel=in_channel,
            num_filters=out_channel,
            kernel_sizes=(1, 3, 5),
            dropout=dropout,
        )
        self.ffn = FFN(
            input_dim=out_channel,
            out_dim=num_classes,
            mid_dims=out_channel // 2,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
        embed = self.embedding(input_ids)
        hidden = self.cnn(embed)
        outs = self.ffn(hidden)

        probs, preds = outs.softmax(-1).max(-1)
        results = {"logits": outs, "probs": probs, "preds": preds}
        if labels is not None:
            results["loss"] = self.loss(outs, labels)
        return results


class DummyPLMModel(nn.Module):
    def __init__(
        self,
        plm_filepath: str,
        num_filters: Optional[int] = 300,
        num_classes: Optional[int] = 3,
        dropout: Optional[float] = 0.5,
    ):
        super().__init__()

        self.plm = AutoModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        self.cnn = MultiKernelCNN(
            in_channel=hidden_size,
            num_filters=num_filters,
            kernel_sizes=(1, 3, 5),
            dropout=dropout,
        )
        self.ffn = FFN(
            input_dim=hidden_size,
            out_dim=num_classes,
            mid_dims=hidden_size // 2,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = self.cnn(plm_outs.last_hidden_state)
        outs = self.ffn(hidden)

        probs, preds = outs.softmax(-1).max(-1)
        results = {"logits": outs, "probs": probs, "preds": preds}
        if labels is not None:
            results["loss"] = self.loss(outs, labels)
        return results
