import os

from rex.utils.io import (
    load_csv,
    dump_line_json
)
from rex.utils.position import find_all_positions


def convert_data(dataset_name,
                 sentence_filepath, 
                 sent_relation_filepath):
    sents = load_csv(sentence_filepath,
                     title_row=False,
                     title_keys=["id", "head", "tail", "text"])
    id2sent = {}
    for ins in sents:
        id2sent[ins["id"]] = ins
    sent_labels = load_csv(sent_relation_filepath,
                           title_row=False, title_keys=["id", "relations"])
    final_data = []
    for ins in sent_labels:
        sent_text = id2sent[ins["id"]]["text"]
        d = {
            "id": ins['id'],
            "tokens": list(sent_text),  # char tokenize
            "entities": [],
            "relations": []
        }
        positions = find_all_positions(sent_text, id2sent[ins["id"]]["head"])
        if len(positions) == 0:
            print("warn! data skipped", d)
            continue
        d['entities'].append(["PER", *positions[0]])  # only take the first occurrence
        positions = find_all_positions(sent_text, id2sent[ins["id"]]["tail"])
        if len(positions) == 0:
            print("warn! data skipped", d)
            continue
        d['entities'].append(["PER", *positions[0]])   # only take the first occurrence
        rels = list(set(map(int, ins["relations"].split())))
        d['relations']= [[id2rel[rel], 0, 1] for rel in rels]
        final_data.append(d)

    print(f'len of {dataset_name}:', len(final_data), final_data[:2])
    dump_line_json(final_data, f"formatted/{dataset_name}.linejson")


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)
    rel2ids = load_csv("raw/IPRE/data/relation2id.txt",
                       title_row=False, title_keys=["relation", "id"])
    rel2id = {}
    for ins in rel2ids:
        rel2id[ins["relation"]] = int(ins["id"])
    id2rel = {id_: rel for rel, id_ in rel2id.items()}

    for dn in ["train", "dev", "test"]:
        convert_data(dn, f"raw/IPRE/data/{dn}/sent_{dn}.txt",
                     f"raw/IPRE/data/{dn}/sent_relation_{dn}.txt")
