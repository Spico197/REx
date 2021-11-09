from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rex.modules.ffn import MLP
from rex.utils.span import find_closest_span_pairs, find_closest_span_pairs_with_index


class SubjObjSpan(nn.Module):
    """
    Inputs:
        hidden: (batch_size, seq_len, hidden_size)
        one_subj_head: object golden head with one subject (batch_size, hidden_size)
        one_subj_tail: object golden tail with one subject (batch_size, hidden_size)
    """

    def __init__(self, hidden_size, num_classes, threshold: Optional[float] = 0.5):
        super().__init__()
        self.threshold = threshold
        self.subj_head_ffnn = nn.Linear(hidden_size, 1)
        self.subj_tail_ffnn = nn.Linear(hidden_size, 1)
        self.obj_head_ffnn = nn.Linear(hidden_size, num_classes)
        self.obj_tail_ffnn = nn.Linear(hidden_size, num_classes)

    def get_objs_for_specific_subj(self, subj_head_mapping, subj_tail_mapping, hidden):
        # (batch_size, 1, hidden_size)
        subj_head = torch.matmul(subj_head_mapping, hidden)
        # (batch_size, 1, hidden_size)
        subj_tail = torch.matmul(subj_tail_mapping, hidden)
        # (batch_size, 1, hidden_size)
        sub = (subj_head + subj_tail) / 2
        # (batch_size, seq_len, hidden_size)
        encoded_text = hidden + sub
        # (batch_size, seq_len, num_classes)
        pred_obj_heads = self.obj_head_ffnn(encoded_text)
        # (batch_size, seq_len, num_classes)
        pred_obj_tails = self.obj_tail_ffnn(encoded_text)
        return pred_obj_heads, pred_obj_tails

    def build_mapping(self, subj_heads, subj_tails):
        """
        Build head & tail mapping for predicted subjects,
        for each instance in a batch, for a subject in all
        the predicted subjects, return a single subject
        and its corresponding mappings.
        """
        for subj_head, subj_tail in zip(subj_heads, subj_tails):
            subjs = find_closest_span_pairs(subj_head, subj_tail)
            seq_len = subj_head.shape[0]
            for subj in subjs:
                subj_head_mapping = torch.zeros(seq_len, device=subj_head.device)
                subj_tail_mapping = torch.zeros(seq_len, device=subj_tail.device)
                subj_head_mapping[subj[0]] = 1.0
                subj_tail_mapping[subj[1]] = 1.0
                yield subj, subj_head_mapping, subj_tail_mapping

    def build_batch_mapping(self, subj_head, subj_tail):
        """
        Build head & tail mapping for predicted subjects,
        for each instance in a batch, return all the predicted
        subjects and mappings.
        """
        subjs = find_closest_span_pairs(subj_head, subj_tail)
        seq_len = subj_head.shape[0]
        if len(subjs) > 0:
            subjs_head_mapping = torch.zeros(
                len(subjs), seq_len, device=subj_head.device
            )
            subjs_tail_mapping = torch.zeros(
                len(subjs), seq_len, device=subj_tail.device
            )

            for subj_idx, subj in enumerate(subjs):
                subjs_head_mapping[subj_idx, subj[0]] = 1.0
                subjs_tail_mapping[subj_idx, subj[1]] = 1.0
            return subjs, subjs_head_mapping, subjs_tail_mapping
        else:
            return None, None, None

    def forward(self, hidden, subj_head, subj_tail):
        # subj_head_out, subj_tail_out: (batch_size, seq_len, 1)
        subj_head_out = self.subj_head_ffnn(hidden)
        subj_tail_out = self.subj_tail_ffnn(hidden)
        # subj_head, subj_tail: (batch_size, seq_len)
        # obj_head_out, obj_tail_out: (batch_size, seq_len, num_classes)
        obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(
            subj_head.unsqueeze(1), subj_tail.unsqueeze(1), hidden
        )

        return (
            subj_head_out.squeeze(-1),
            subj_tail_out.squeeze(-1),
            obj_head_out,
            obj_tail_out,
        )

    def predict(self, hidden):
        # hidden: 1 x hidden_size
        if hidden.shape[0] != 1:
            raise RuntimeError(
                (
                    "eval batch size must be 1 x hidden_size, "
                    f"while hidden is {hidden.shape}"
                )
            )
        # 1, hidden_size, 1
        subj_head_out = self.subj_head_ffnn(hidden)
        # 1, hidden_size, 1
        subj_tail_out = self.subj_tail_ffnn(hidden)
        subj_head_out = torch.sigmoid(subj_head_out)
        subj_tail_out = torch.sigmoid(subj_tail_out)
        pred_subj_head = subj_head_out.ge(self.threshold).long()
        pred_subj_tail = subj_tail_out.ge(self.threshold).long()

        triples = []
        # subjs: [(1, 3), (3, 3), (17, 17)]
        # subj_head_mappings: (len(subjs), seq_len)
        subjs, subj_head_mappings, subj_tail_mappings = self.build_batch_mapping(
            pred_subj_head.squeeze(0).squeeze(-1), pred_subj_tail.squeeze(0).squeeze(-1)
        )

        if subjs:
            # obj_head_out: (len(subjs), seq_len, num_classes)
            obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(
                subj_head_mappings.unsqueeze(1), subj_tail_mappings.unsqueeze(1), hidden
            )
            obj_head_out = torch.sigmoid(obj_head_out)
            obj_tail_out = torch.sigmoid(obj_tail_out)
            obj_head_out = obj_head_out.ge(self.threshold).long()
            obj_tail_out = obj_tail_out.ge(self.threshold).long()
            for subj_idx, subj in enumerate(subjs):
                objs = find_closest_span_pairs_with_index(
                    obj_head_out[subj_idx].permute(1, 0),
                    obj_tail_out[subj_idx].permute(1, 0),
                )
                for relation_idx, obj_pair_start, obj_pair_end in objs:
                    triples.append(
                        (
                            (subj[0], subj[1] + 1),
                            relation_idx,
                            (obj_pair_start, obj_pair_end + 1),
                        )
                    )
        return [triples]
