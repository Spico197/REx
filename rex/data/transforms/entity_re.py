import random
from collections import defaultdict
from typing import Iterable, List, Dict, Optional

import numpy as np
from transformers.models.bert import BertTokenizerFast

from rex.data.vocab import Vocab
from rex.data.label_encoder import LabelEncoder
from rex.data.transforms.base import TransformBase


class StreamSubjObjSpanTransform(TransformBase):
    """
    Data transform for jointly entity-relation extraction task.
    """

    def __init__(
        self, max_seq_len: int, rel2id: Dict, vocab_filepath: str, pad="PAD", unk="UNK"
    ) -> None:
        super().__init__(max_seq_len)
        self.vocab = Vocab.from_pretrained(vocab_filepath, pad=pad, unk=unk)
        self.label_encoder = LabelEncoder(rel2id)

    def transform(self, data: Iterable) -> List[dict]:
        token_ids, mask = self.vocab.encode(
            data["tokens"], self.max_seq_len, update=False
        )

        subj_heads = np.zeros(self.max_seq_len, dtype=np.uint8)
        subj_tails = np.zeros(self.max_seq_len, dtype=np.uint8)
        one_subj_head = np.zeros(self.max_seq_len, dtype=np.uint8)
        one_subj_tail = np.zeros(self.max_seq_len, dtype=np.uint8)
        all_obj_head = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        all_obj_tail = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        one_obj_head = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        one_obj_tail = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )

        subj2objs = defaultdict(list)
        triples = []
        for relation_triple in data["relations"]:
            relation = self.label_encoder.encode([relation_triple[0]])[0]
            head_ent = data["entities"][relation_triple[1]]
            tail_ent = data["entities"][relation_triple[2]]
            triples.append(
                ((head_ent[1], head_ent[2]), relation, (tail_ent[1], tail_ent[2]))
            )
            if head_ent[2] <= self.max_seq_len:
                subj_heads[head_ent[1]] = 1
                subj_tails[head_ent[2] - 1] = 1
                subj2objs[(head_ent[1], head_ent[2])].append(
                    (relation, tail_ent[1], tail_ent[2])
                )
            if tail_ent[2] <= self.max_seq_len:
                all_obj_head[relation, tail_ent[1]] = 1
                all_obj_tail[relation, tail_ent[2] - 1] = 1

        one_subj = random.choice(list(subj2objs.keys()))
        one_subj_head[one_subj[0]] = 1
        one_subj_tail[one_subj[1] - 1] = 1
        for obj in subj2objs[one_subj]:
            one_obj_head[obj[0], obj[1]] = 1
            one_obj_tail[obj[0], obj[2] - 1] = 1

        formatted = {
            "token_ids": token_ids,
            "mask": mask,
            "subj_heads": subj_heads,
            "subj_tails": subj_tails,
            "one_subj": one_subj,
            "subj2objs": subj2objs,
            "triples": triples,
            "subj_head": one_subj_head,
            "subj_tail": one_subj_tail,
            "obj_head": one_obj_head.transpose(),
            "obj_tail": one_obj_tail.transpose(),
        }
        formatted.update(data)
        return formatted

    def predict_transform(self, obj: dict):
        """
        Args:
            obj:
                {
                    "text": "text"  # space tokenized string
                }
        """
        obj["text"] = obj["text"].split()
        token_ids, mask = self.vocab.encode(
            list(obj["text"]), self.max_seq_len, update=False
        )
        d = {
            "token_ids": token_ids,
            "mask": mask,
        }
        return d


class StreamBERTSubjObjSpanTransform(TransformBase):
    """
    Data transform for jointly entity-relation extraction task, with BERT tokenizers.
    """

    def __init__(self, max_seq_len: int, rel2id: Dict, bert_model_dir: str) -> None:
        super().__init__(max_seq_len)
        self.vocab = None
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_dir)
        self.label_encoder = LabelEncoder(rel2id)

    def tokenize(self, tokens: List[str]):
        token_list = []
        for token in tokens:
            new_token = self.tokenizer.tokenize(token)
            token_list.append(new_token)
        return token_list

    def encode(self, token_list: List[List], max_seq_len: int):
        final_tokens = [self.tokenizer.cls_token]
        final_mask = [1]
        flat_tokens = []
        for token in token_list:
            flat_tokens.extend(token)
        flat_tokens = flat_tokens[: max_seq_len - 2]
        final_tokens.extend(flat_tokens)
        final_tokens.append(self.tokenizer.sep_token)
        final_mask += [1] * len(flat_tokens)
        final_mask.append(1)
        assert len(final_tokens) == len(final_mask)
        assert len(final_mask) <= max_seq_len
        final_tokens += [self.tokenizer.pad_token] * (max_seq_len - len(final_tokens))
        final_tokens = self.tokenizer.convert_tokens_to_ids(final_tokens)
        final_mask += [0] * (max_seq_len - len(final_mask))

        return final_tokens, final_mask

    def get_offset_position(
        self, token_list: List[List], pos: int, offset: Optional[int] = 1
    ):
        """
        get new position after tokenized

        Args:
            offset: if [CLS] is used at the beginning of a sequence, offset must be 1
        """
        new_pos = -1
        record_idx = 0
        for idx, tokens in enumerate(token_list):
            if idx == pos:
                new_pos = record_idx
                break
            else:
                record_idx += len(tokens)
        return new_pos

    def transform(self, data: Iterable) -> List[dict]:
        token_list = self.tokenize(data["tokens"])
        token_ids, mask = self.encode(token_list, self.max_seq_len)

        subj_heads = np.zeros(self.max_seq_len, dtype=np.uint8)
        subj_tails = np.zeros(self.max_seq_len, dtype=np.uint8)
        one_subj_head = np.zeros(self.max_seq_len, dtype=np.uint8)
        one_subj_tail = np.zeros(self.max_seq_len, dtype=np.uint8)
        all_obj_head = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        all_obj_tail = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        one_obj_head = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )
        one_obj_tail = np.zeros(
            (self.label_encoder.num_tags, self.max_seq_len), dtype=np.uint8
        )

        subj2objs = defaultdict(list)
        triples = []
        for relation_triple in data["relations"]:
            relation = self.label_encoder.encode([relation_triple[0]])[0]
            head_ent = data["entities"][relation_triple[1]]
            tail_ent = data["entities"][relation_triple[2]]
            head_ent_start = self.get_offset_position(token_list, head_ent[1])
            head_ent_end = self.get_offset_position(token_list, head_ent[2])
            tail_ent_start = self.get_offset_position(token_list, tail_ent[1])
            tail_ent_end = self.get_offset_position(token_list, tail_ent[2])
            if (
                0 < head_ent_start < self.max_seq_len
                and 0 < head_ent_end < self.max_seq_len
                and 0 < tail_ent_start < self.max_seq_len
                and 0 < tail_ent_end < self.max_seq_len
            ):
                triples.append(
                    (
                        (head_ent_start, head_ent_end),
                        relation,
                        (tail_ent_start, tail_ent_end),
                    )
                )
                subj_heads[head_ent_start] = 1
                subj_tails[head_ent_end - 1] = 1
                subj2objs[(head_ent_start, head_ent_end)].append(
                    (relation, tail_ent_start, tail_ent_end)
                )
                all_obj_head[relation, tail_ent_start] = 1
                all_obj_tail[relation, tail_ent_end - 1] = 1

        one_subj = None
        if len(subj2objs) > 0:
            one_subj = random.choice(list(subj2objs.keys()))
            one_subj_head[one_subj[0]] = 1
            one_subj_tail[one_subj[1] - 1] = 1
            for obj in subj2objs[one_subj]:
                one_obj_head[obj[0], obj[1]] = 1
                one_obj_tail[obj[0], obj[2] - 1] = 1

        formatted = {
            "token_ids": token_ids,
            "mask": mask,
            "subj_heads": subj_heads,
            "subj_tails": subj_tails,
            "one_subj": one_subj,
            "subj2objs": subj2objs,
            "triples": triples,
            "subj_head": one_subj_head,
            "subj_tail": one_subj_tail,
            "obj_head": one_obj_head.transpose(),
            "obj_tail": one_obj_tail.transpose(),
        }
        formatted.update(data)
        return formatted

    def predict_transform(self, obj: dict):
        """
        Args:
            obj:
                {
                    "text": "text"  # space tokenized string
                }
        """
        token_list = self.tokenize(obj["text"].split())
        token_ids, mask = self.encode(token_list, max_seq_len=self.max_seq_len)
        d = {
            "token_ids": token_ids,
            "mask": mask,
        }
        return d
