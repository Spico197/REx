from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from rex.data.transforms.base import CachedTransformOneBase
from rex.data.collate_fn import GeneralCollateFn


class MaxLengthExceedException(Exception):
    pass


class USMTransform(CachedTransformOneBase):
    def __init__(self, plm_dir: str, max_seq_len: int = 512) -> None:
        super().__init__()

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(plm_dir)
        self.lm_token = "[LM]"
        self.lp_token = "[LP]"
        self.text_token = "[T]"
        num_added = self.tokenizer.add_tokens([self.lm_token, self.lp_token, self.text_token], special_tokens=True)
        assert num_added == 3
        self.lm_token_id, self.lp_token_id, self.text_token_id = self.tokenizer.convert_tokens_to_ids([self.lm_token, self.lp_token, self.text_token])

        self.max_seq_len = max_seq_len

    def build_label_seq(self, ent_labels: set, relation_labels: set) -> tuple:
        """
        Returns:
            input_ids
            input_tokens
            mask
            label_map: {label index: {"type": "m"/"p", "string": "person"}, ...}
            label_str_to_idx: {("person", "m"): label index}
        """
        label_map = {}
        label_str_to_idx = {}
        mask = [1]
        input_tokens = [self.tokenizer.cls_token]
        for ent in ent_labels:
            input_tokens.append(self.lm_token)
            mask.append(2)
            label_index = len(input_tokens)
            label_map[label_index] = {"type": "m", "string": ent}
            label_str_to_idx[(ent, "m")] = label_index
            label_tokens = self.tokenizer.tokenize(ent)
            input_tokens.extend(label_tokens)
            mask.extend([3] * len(label_tokens))
        for rel in relation_labels:
            input_tokens.append(self.lp_token)
            mask.append(4)
            label_index = len(input_tokens)
            label_map[label_index] = {"type": "p", "string": rel}
            label_str_to_idx[(rel, "p")] = label_index
            label_tokens = self.tokenizer.tokenize(rel)
            input_tokens.extend(label_tokens)
            mask.extend([5] * len(label_tokens))
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_ids, input_tokens, mask, label_map, label_str_to_idx

    def build_input_seq(self, tokens: list, ent_labels: set, rel_labels: set):
        input_ids, input_tokens, mask, label_map, label_str_to_idx = self.build_label_seq(ent_labels, rel_labels)
        input_tokens.append(self.text_token)
        mask.append(6)
        offset = len(mask)
        remain_len = self.max_seq_len - offset - 1
        if remain_len <= 0:
            raise MaxLengthExceedException
        remain_tokens = tokens[:remain_len]
        input_tokens.extend(remain_tokens)
        mask.extend([7] * remain_len)
        input_tokens.append(self.tokenizer.sep_token)
        mask.append(8)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_ids, input_tokens, mask, label_map, label_str_to_idx, offset

    def transform(self, instance: dict, **kwargs) -> dict:
        """
        Args:
            instance: {
                "id": "idx".
                "tokens": ["token", "##1"],
                "ents": [[[start, end + 1], "label"], ...],
                "relations": [[[head start, head end + 1], "relation", [tail start, tail end + 1]], ...],
                "events": [{"event_type": "event type", "trigger": [start, end + 1], "arguments": [[[arg start, end + 1], "role"], ...]}, ...],
            }
        """
        ent_labels = set(x[1] for x in instance["ents"])
        ent_labels.update(x["event_type"] for x in instance["events"])
        rel_labels = set(x[1] for x in instance["relations"])
        rel_labels.update(x[1] for e in instance["events"] for x in e["arguments"])
        # TODO
        input_ids, input_tokens, mask, label_map, label_str_to_idx = self.build_label_seq(ent_labels, rel_labels)


        offset = len(input_ids)
        return instance

    def predict_transform(self, *args, **kwargs) -> dict:
        return super().predict_transform(*args, **kwargs)
