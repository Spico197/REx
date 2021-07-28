from typing import Dict, Iterable, Optional, List, Union


class LabelEncoder(object):
    def __init__(self, initial_dict: Optional[Dict] = {}) -> None:
        self.label2id = initial_dict
        self.id2label = {idx: label for idx, label in enumerate(self.label2id)}

    def add(self, label: Union[str, int]):
        if label not in self.label2id:
            label_id = len(self.label2id)
            self.label2id[label] = label_id
            self.id2label[label_id] = label

    def update(self, labels):
        for label in labels:
            self.add(label)

    def encode(self, labels: List[Union[str, int]]):
        if not all(label in self.label2id for label in labels):
            raise ValueError("Not all label are in this encoder")
        return list(map(lambda x: self.label2id[x], labels))

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

    @property
    def num_tags(self):
        return len(self)
