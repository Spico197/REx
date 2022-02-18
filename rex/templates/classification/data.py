from typing import Iterable, Optional, List

import torch
from rex.utils.logging import logger
from rex.utils.progress_bar import tqdm
from rex.data.label_encoder import LabelEncoder
from rex.data.transforms.base import TransformBase


def mc_collate_fn(data):
    final_data = {
        "id": [],
        "input_ids": [],
        "mask": [],
        "labels": [],
    }
    for d in data:
        for key in final_data:
            final_data[key].append(d[key])

    final_data["token_ids"] = torch.tensor(final_data["token_ids"], dtype=torch.long)
    final_data["mask"] = torch.tensor(final_data["mask"], dtype=torch.long)
    if all(x is not None for x in final_data["labels"]):
        final_data["labels"] = torch.tensor(final_data["labels"], dtype=torch.long)
    else:
        final_data["labels"] = None

    return final_data


class CachedMCTransform(TransformBase):
    """Cached data transform for classification task."""

    def __init__(self, max_seq_len: int) -> None:
        super().__init__(max_seq_len)

        self.label_encoder = LabelEncoder()

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
        transform_loader = tqdm(dataset, desc=desc)

        num_tot_ins = 0
        for data in transform_loader:
            label = data["label"]
            label_id = self.label_encoder.update_encode_one(label)
            text = data["text"]
            token_ids, mask = self.vocab.encode(
                list(text), max_seq_len=self.max_seq_len, update=True
            )

            ins = {
                "id": data["id"],
                "input_ids": token_ids,
                "mask": mask,
                "labels": label_id,
                "text": text,
            }
            final_data.append(ins)

        logger.info(transform_loader)
        logger.info(f"#Total Ins: {num_tot_ins}")

        return final_data

    def predict_transform(self, data: dict):
        """
        Args:
            data:
                {
                    "id": "blahblah",
                    "text": "text",
                }
        """
        text = data["text"]
        token_ids, mask = self.vocab.encode(
            list(text), max_seq_len=self.max_seq_len, update=False
        )

        obj = {
            "input_ids": token_ids,
            "mask": mask,
        }
        return obj
