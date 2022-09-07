import json
import os
from typing import List

from rex.utils.build_emb import build_emb
from rex.utils.io import dump_json, dump_jsonlines, load_json, load_jsonlines_iterator
from rex.utils.position import find_all_positions


def convert_data(dataset_name, filepath):
    final_data = []
    for idx, ins in enumerate(load_jsonlines_iterator(filepath)):
        d = {
            "id": str(idx),
            "tokens": list(ins["text"].lower().split()),
            "entities": [],
            "relations": [],
        }
        positions = find_all_positions(d["tokens"], ins["h"]["name"].lower().split())
        if len(positions) == 0:
            print("warn! data skipped", ins)
            continue
        d["entities"].append(["O", *positions[0]])  # only take the first occurrence
        positions = find_all_positions(d["tokens"], ins["t"]["name"].lower().split())
        if len(positions) == 0:
            print("warn! data skipped", ins)
            continue
        d["entities"].append(["O", *positions[0]])  # only take the first occurrence
        d["relations"] = [[ins["relation"], 0, 1]]
        final_data.append(d)

    print(f"len of {dataset_name}:", len(final_data), final_data[:2])
    dump_jsonlines(final_data, f"formatted/{dataset_name}.jsonl")


def _tokenize(line: str) -> List[str]:
    ins = json.loads(line.strip())
    tokens = [token.lower() for token in ins["tokens"]]
    return tokens


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)
    dump_json(load_json("raw/nyt10_rel2id.json"), "formatted/rel2id.json")
    for dn in ["train", "test"]:
        convert_data(dn, f"raw/nyt10_{dn}.txt")

    build_emb(
        "/data/tzhu/data/glove/glove.6B.300d.txt",
        "formatted/vocab.emb",
        "formatted/train.jsonl",
        "formatted/test.jsonl",
        tokenize_func=_tokenize,
    )
