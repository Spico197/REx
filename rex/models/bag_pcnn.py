from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rex.modules.pcnn import PiecewiseCNN
from rex.modules.embeddings.static_embedding import StaticEmbedding


class PCNNOne(nn.Module):
    """
    PCNN + ONE for bag-level multi-class relation classification

    References:
        - Zeng D et al. Distant supervision for relation extraction via piecewise convolutional neural networks. EMNLP. 2015.
    """

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

    def forward(self, token_ids, head_pos, tail_pos, mask, scopes, labels=None):
        token_embedding = self.token_embedding(token_ids)
        pos1_embedding = self.pos1_embedding(head_pos)
        pos2_embedding = self.pos2_embedding(tail_pos)
        x = torch.cat([token_embedding, pos1_embedding, pos2_embedding], dim=-1)
        x = self.pcnn(x, mask)

        if labels is not None:
            bag_rep_for_loss = []
        bag_rep_for_pred = []
        for ind, scope in enumerate(scopes):
            sent_rep = x[scope[0] : scope[1], :]
            out = self.dense(sent_rep)
            probs = torch.softmax(out, dim=-1)

            # for training loss calculation
            if labels is not None:
                _, j = torch.max(probs[:, labels[ind]], dim=-1)
                bag_rep_for_loss.append(out[j])

            # for prediction
            row_prob, row_idx = torch.max(probs, dim=-1)
            if row_idx.sum() > 0:
                mask = row_idx.view(-1, 1).expand(-1, probs.shape[-1])
                probs = probs.masked_fill_(mask.eq(0), float("-inf"))
                row_prob, _ = torch.max(probs[:, 1:], dim=-1)
                _, row_idx = torch.max(row_prob, dim=0)
            else:
                _, row_idx = torch.max(row_prob, dim=-1)
            bag_rep_for_pred.append(out[row_idx])

        result = {}
        if labels is not None:
            bag_rep_for_loss = torch.stack(bag_rep_for_loss, dim=0)
            result.update({"loss": F.cross_entropy(bag_rep_for_loss, labels)})
        bag_rep_for_pred = torch.stack(bag_rep_for_pred, dim=0)
        result.update({"pred": bag_rep_for_pred.argmax(dim=-1)})

        return result


class PCNNAtt(nn.Module):
    """
    PCNN + Selective Attention for bag-level multi-class relation classification

    References:
        - Lin Y et al. Neural relation extraction with selective attention over instances. ACL. 2016.
    """

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, head_pos, tail_pos, mask, scopes, labels=None):
        token_embedding = self.token_embedding(token_ids)
        pos1_embedding = self.pos1_embedding(head_pos)
        pos2_embedding = self.pos2_embedding(tail_pos)
        x = torch.cat([token_embedding, pos1_embedding, pos2_embedding], dim=-1)
        x = self.pcnn(x, mask)

        if labels is not None:
            bag_rep_for_loss = []
        bag_rep_for_pred = []

        # for training loss calculation
        if labels is not None:
            query = torch.zeros((x.size(0)), device=x.device, dtype=torch.long)
            for i in range(len(scopes)):
                query[scopes[i][0] : scopes[i][1]] = labels[i]
            att_mat = self.dense.weight[query]
            att_score = (x * att_mat).sum(-1)
            bag_rep_for_loss = []
            for i in range(len(scopes)):
                bag_mat = x[scopes[i][0] : scopes[i][1]]
                softmax_att_score = F.softmax(
                    att_score[scopes[i][0] : scopes[i][1]], dim=-1
                )
                bag_rep_for_loss.append(
                    (softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)
                )
            bag_rep_for_loss = torch.stack(bag_rep_for_loss, 0)
            bag_rep_for_loss = self.dropout(bag_rep_for_loss)
            bag_rep_for_loss = self.dense(bag_rep_for_loss)

        # for prediction
        att_score = torch.matmul(x, self.dense.weight.transpose(0, 1))
        for i in range(len(scopes)):
            bag_mat = x[scopes[i][0] : scopes[i][1]]
            softmax_att_score = F.softmax(
                att_score[scopes[i][0] : scopes[i][1]].transpose(0, 1), dim=-1
            )
            rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)
            logit_for_each_rel = F.softmax(self.dense(rep_for_each_rel), dim=-1)
            logit_for_each_rel = logit_for_each_rel.diag()
            bag_rep_for_pred.append(logit_for_each_rel)
        bag_rep_for_pred = torch.stack(bag_rep_for_pred, 0)

        result = {}
        if labels is not None:
            result.update({"loss": F.cross_entropy(bag_rep_for_loss, labels)})
        result.update({"pred": bag_rep_for_pred.argmax(dim=-1)})

        return result
