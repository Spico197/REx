from typing import Iterable, List, Optional, Union

from rex.utils.io import dump_json, load_json


class LabelEncoder(object):
    def __init__(self, initial: Optional[Union[dict, Iterable]] = None) -> None:
        self.label2id = {}
        self.id2label = {}

        if initial is not None:
            if isinstance(initial, dict):
                _pairs = tuple(initial.items())
                self.label2id = {key: val for key, val in _pairs}
                self.id2label = {idx: label for idx, label in enumerate(self.label2id)}
            elif isinstance(initial, Iterable):
                self.update(initial)
            else:
                raise ValueError(f"Type {type(initial)} not supported!")

    def add(self, label: Union[str, int]):
        if label not in self.label2id:
            label_id = len(self.label2id)
            self.label2id[label] = label_id
            self.id2label[label_id] = label

    def update(self, labels):
        for label in labels:
            self.add(label)

    def encode(self, labels: List[Union[str, int]], update: Optional[bool] = False):
        if update:
            self.update(labels)

        if not all(label in self.label2id for label in labels):
            raise ValueError("Not all label are in this encoder")

        return list(map(lambda x: self.label2id[x], labels))

    def encode_one(self, label, update: bool = False):
        return self.encode([label], update=update)[0]

    def decode_one(self, label_idx):
        return self.decode([label_idx])[0]

    def update_encode(self, labels):
        self.update(labels)
        return self.encode(labels)

    def update_encode_one(self, label):
        self.add(label)
        return self.encode([label])[0]

    def decode(self, ids: List[int]):
        if not all(idx in self.id2label for idx in ids):
            raise ValueError("Not all idx are in this encoder")
        return list(map(lambda x: self.id2label[x], ids))

    def convert_to_multi_hot(self, label_ids: Iterable[int]):
        labels = [0] * self.num_tags
        for label_id in label_ids:
            labels[label_id] = 1
        return labels

    def convert_to_one_hot(self, label_id: int):
        labels = [0] * self.num_tags
        labels[label_id] = 1
        return labels

    def __len__(self):
        return len(self.label2id)

    def __contains__(self, item):
        return item in self.label2id

    @property
    def num_tags(self):
        return len(self)

    @classmethod
    def from_pretrained(cls, filepath: str):
        label2id = load_json(filepath)
        return cls(initial=label2id)

    def save_pretrained(self, filepath: str):
        dump_json(self.label2id, filepath)
