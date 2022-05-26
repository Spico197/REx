from typing import Iterable, List, Optional

import torch
from transformers import BertTokenizerFast

from rex.data.collate_fn import GeneralCollateFn
from rex.data.label_encoder import LabelEncoder
from rex.data.transforms.base import TransformBase
from rex.data.vocab import Vocab, get_pad_mask, get_pad_token
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar


class CachedTaggingTransform(TransformBase):
    """Cached data transform for classification task."""

    def __init__(self, max_seq_len: int, label2id_filepath: str, plm_dir: str) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.label_encoder = LabelEncoder.from_pretrained(label2id_filepath)
        self.tokenizer = BertTokenizerFast.from_pretrained(plm_dir)

        self.collate_fn = GeneralCollateFn(
            {
                # None means keep its original type
                "id": None,
                "tokens": None,
                "ner_tags": None,
                "token_ids": torch.long,
                "mask": torch.long,
                "labels": torch.long,
            }
        )

    def transform(
        self,
        dataset: Iterable,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        **kwargs,
    ) -> List[dict]:
        final_data = []
        if debug:
            dataset = dataset[:500]
        transform_loader = pbar(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            tokens = data["tokens"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = get_pad_token(
                token_ids, self.max_seq_len, self.tokenizer.pad_token_id
            )
            mask = get_pad_mask(len(token_ids), self.max_seq_len)
            labels = data["ner_tags"]
            label_ids = self.label_encoder.encode(labels, update=False)
            label_ids = get_pad_token(
                label_ids, self.max_seq_len, self.label_encoder.label2id["O"]
            )

            assert len(token_ids) == self.max_seq_len
            assert len(mask) == self.max_seq_len
            assert len(label_ids) == self.max_seq_len

            ins = {
                "id": data["id"],
                "tokens": data["tokens"],
                "ner_tags": data["ner_tags"],
                "token_ids": token_ids,
                "mask": mask,
                "labels": label_ids,
            }
            final_data.append(ins)

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def predict_transform(self, text: str):
        """
        Args:
            data:
                {
                    "id": "blahblah",
                    "text": "text",
                }
        """
        token_ids = self.tokenizer.convert_tokens_to_ids(list(text))
        token_ids = get_pad_token(token_ids, self.max_seq_len, self.tokenizer.pad)
        mask = get_pad_mask(len(token_ids), self.max_seq_len)

        obj = {
            "tokens": list(text),
            "token_ids": token_ids,
            "mask": mask,
        }
        return obj
