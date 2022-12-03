from typing import Any, Dict, List, Optional

import torch

from rex.utils.batch import (
    group_instances_into_batch,
    validate_instance_has_the_same_keys,
)
from rex.utils.iteration import flatten_all


class GeneralCollateFn(object):
    """DataLoader collate function for general purpose
        If you want to transform data while collating,
        inherent this class and override the ``update_data`` method

    Args:
        key2type (~dict): For type mapping initialization. When collating,
            only values whose keys are in ``key2type`` are grouped together,
            the others will be dropped.
        guessing (~bool): Guess type for each field. We recommend you
            to write your own ``key2type`` mapping when instantiate
            this collate obj and set ``guessing`` to False in case of
            any problems.
    """

    DEFAULT_TYPE_MAP = {int: torch.long, float: torch.float, str: None}

    def __init__(
        self,
        key2type: Optional[Dict[str, Any]],
        guessing: Optional[bool] = False,
        missing_key_as_null: Optional[bool] = False,
    ) -> None:
        self.key2type = {}
        if isinstance(key2type, dict) and len(key2type) > 0:
            self.key2type.update(key2type)
        self.guessing = guessing
        self.missing_key_as_null = missing_key_as_null

    def update_type_mapping(self, key2type: dict):
        for key, val_type in key2type.items():
            self.key2type[key] = val_type

    def guess_types(self, instance: dict, update: Optional[bool] = False) -> dict:
        key2type = {}
        for key, val in instance.items():
            val_type = type(val)
            if val_type in self.DEFAULT_TYPE_MAP:
                key2type[key] = self.DEFAULT_TYPE_MAP[val_type]
            elif val_type == list:
                item_types = set()
                for item in flatten_all(val):
                    item_types.add(type(item))
                if len(item_types) == 1:
                    key2type[key] = self.DEFAULT_TYPE_MAP.get(item_types.pop())
                else:
                    key2type[key] = None
            else:
                key2type[key] = None

        if update:
            self.update_type_mapping(key2type)
        return key2type

    def update_data(self, data: List[dict]) -> List[dict]:
        """For those who transform data while collating, override this function"""
        return data

    def update_before_tensorify(self, data: dict):
        return data

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        validate_instance_has_the_same_keys(data)
        if len(self.key2type) == 0 and self.guessing:
            self.guess_types(data[0], update=True)

        if self.missing_key_as_null:
            for d in data:
                missing_keys = set(d.keys()) - set(self.key2type.keys())
                for mkey in missing_keys:
                    self.key2type[mkey] = None

        data = self.update_data(data)
        final_data = group_instances_into_batch(data, self.key2type.keys())
        final_data = self.update_before_tensorify(final_data)

        for key, val_type in self.key2type.items():
            if val_type is not None and all(val is not None for val in final_data[key]):
                final_data[key] = torch.tensor(final_data[key], dtype=val_type)

        return final_data


def re_collate_fn(data):
    final_data = {
        "id": [],
        "token_ids": [],
        "mask": [],
        "labels": [],
        "head_pos": [],
        "tail_pos": [],
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
    final_data["head_pos"] = torch.tensor(final_data["head_pos"], dtype=torch.long)
    final_data["tail_pos"] = torch.tensor(final_data["tail_pos"], dtype=torch.long)

    return final_data


def bag_re_collate_fn(data):
    final_data = {
        "id": [],
        "token_ids": [],
        "mask": [],
        "labels": [],
        "head_pos": [],
        "tail_pos": [],
        "scopes": [],
    }
    scope = []
    scope_begin = 0
    labels = []
    # data is a batch, for every bag in the batch:
    for bag in data:
        scope_ = [scope_begin]
        # for every instance in the bag
        for ins in bag:
            for key in final_data:
                if key != "scopes":
                    final_data[key].append(ins[key])
            scope_begin += 1
        labels.append(ins["labels"])
        scope_.append(scope_begin)
        scope.append(scope_)
    final_data["scopes"] = scope

    final_data["token_ids"] = torch.tensor(final_data["token_ids"], dtype=torch.long)
    final_data["mask"] = torch.tensor(final_data["mask"], dtype=torch.long)
    if all(x is not None for x in final_data["labels"]):
        final_data["labels"] = torch.tensor(labels, dtype=torch.long)
    else:
        final_data["labels"] = None
    final_data["head_pos"] = torch.tensor(final_data["head_pos"], dtype=torch.long)
    final_data["tail_pos"] = torch.tensor(final_data["tail_pos"], dtype=torch.long)

    return final_data


def subj_obj_span_collate_fn(data):
    final_data = {
        "id": [],
        "tokens": [],
        "entities": [],
        "relations": [],
        "token_ids": [],
        "mask": [],
        "subj_heads": [],
        "subj_tails": [],
        "one_subj": [],
        "subj2objs": [],
        "triples": [],
        "subj_head": [],
        "subj_tail": [],
        "obj_head": [],
        "obj_tail": [],
    }
    for d in data:
        for key in final_data:
            if key in d:
                final_data[key].append(d[key])

    final_data["token_ids"] = torch.tensor(final_data["token_ids"], dtype=torch.long)
    final_data["mask"] = torch.tensor(final_data["mask"], dtype=torch.long)
    for key in [
        "subj_heads",
        "subj_tails",
        "subj_head",
        "subj_tail",
        "obj_head",
        "obj_tail",
    ]:
        if final_data[key]:
            final_data[key] = torch.tensor(final_data[key], dtype=torch.float)

    return final_data
