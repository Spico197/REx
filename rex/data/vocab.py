from typing import Iterable, Optional, List, Tuple

from rex.utils.io import load_line_iterator, dump_iterable


class Vocab(object):
    def __init__(
        self, pad: Optional[str] = "[PAD]", unk: Optional[str] = "[UNK]"
    ) -> None:
        self.pad = pad
        self.unk = unk
        self.token2id = {pad: 0, unk: 1}
        self.pad_idx = self.token2id[pad]
        self.unk_idx = self.token2id[unk]

        self.id2token = {self.pad_idx: pad, self.unk_idx: unk}

    def add(self, token: str):
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token

    def update(self, tokens: List[str]):
        for token in tokens:
            self.add(token)

    def convert_tokens_to_ids(self, tokens: List[str]):
        return list(map(lambda x: self.token2id.get(x, self.unk_idx), tokens))

    def convert_ids_to_tokens(self, ids: List[int]):
        if not all(idx in self.id2token for idx in ids):
            raise ValueError("Not all idx are in this vocab")
        return list(map(lambda x: self.id2token[x], ids))

    def update_convert_tokens_to_ids(self, tokens: List[str]):
        for token in tokens:
            self.add(token)
        return self.convert_tokens_to_ids(tokens)

    def encode(
        self, tokens: Iterable, max_seq_len: int, update: Optional[bool] = False
    ) -> Tuple[List[int]]:
        """convert tokens into ids by padding or cutting

        Args:
            update: whether to add tokens into vocab

        Returns:
            token_ids: token ids after encoding
            mask: padding mask
        """
        tokens = tokens[:max_seq_len]
        mask = [1] * len(tokens) + [0] * (max_seq_len - len(tokens))
        tokens = tokens + [self.pad] * (max_seq_len - len(tokens))
        if update:
            return self.update_convert_tokens_to_ids(tokens), mask
        else:
            return self.convert_tokens_to_ids(tokens), mask

    def clear_all(self):
        self.token2id.clear()
        self.id2token.clear()

    def __len__(self):
        return len(self.token2id)

    @property
    def size(self):
        return len(self)

    @classmethod
    def from_pretrained(
        cls, filepath, pad: Optional[str] = "[PAD]", unk: Optional[str] = "[UNK]"
    ):
        v = cls(pad, unk)
        for line in load_line_iterator(filepath):
            v.add(line.strip())
        return v

    def save_pretrained(self, filepath):
        vocabs = []
        for token_id in range(self.size):
            vocabs.append(self.id2token[token_id])
        dump_iterable(vocabs, filepath)
