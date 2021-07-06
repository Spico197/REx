import os
from collections import Counter

from rex.utils.io import (
    dump_json,
    load_json,
    dump_line_json
)
from rex.utils.position import find_all_positions


def convert_data(dataset_name, filepath):
    data = load_json(filepath)
    final_data = []
    lens = []
    for ins_idx, ins in enumerate(data):
        ins['text'] = ins['text'].replace(' ', '')
        tokens = list(ins['text'])
        lens.append(len(tokens))
        d = {
            "id": f"{dataset_name.upper()}.{ins_idx}",
            "tokens": tokens,  # char tokenize
            "entities": [],
            "relations": []
        }
        for head, rel, tail in ins['triple_list']:
            head = head.replace(' ', '')
            tail = tail.replace(' ', '')
            try:
                head_pos = find_all_positions(tokens, list(head))
                tail_pos = find_all_positions(tokens, list(tail))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception:
                continue
            if not head_pos or not tail_pos:
                continue
            head_pos = head_pos[0]
            tail_pos = tail_pos[0]
            head_ent = ["ENTITY", *head_pos, head]
            if head_ent not in d['entities']:
                d['entities'].append(head_ent)
            tail_ent = ["ENTITY", *tail_pos, tail]
            if tail_ent not in d['entities']:
                d['entities'].append(tail_ent)
            relation = rel
            rels.add(relation)
            d['relations'].append([
                relation, d['entities'].index(head_ent), d['entities'].index(tail_ent),
                [head_ent[3], tail_ent[3]]
            ])
        final_data.append(d)

    print(f'len of {dataset_name}:', len(final_data), final_data[:2])
    dump_line_json(final_data, f"formatted/{dataset_name}.linejson")

    len_counter = Counter(lens)
    print(max(lens), len_counter.most_common())


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)

    rels = set()

    for dn in ["train", "dev", "test"]:
        convert_data(dn, f"raw/{dn}_triples.json")

    dump_json({rel: idx for idx, rel in enumerate(rels)}, "formatted/rel2id.json", indent=2)
