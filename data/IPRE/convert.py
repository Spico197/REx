import os
import json

from rex.io.utils import load_csv, dump_line_json, dump_json


if __name__ == "__main__":
    os.makedirs("formatted/bag", exist_ok=True)
    os.makedirs("formatted/sent", exist_ok=True)

    rel2ids = load_csv("raw/IPRE/data/relation2id.txt",
                       title_row=False, title_keys=["relation", "id"])
    rel2id = {}
    for ins in rel2ids:
        rel2id[ins["relation"]] = int(ins["id"])
    id2rel = {id_: rel for rel, id_ in rel2id.items()}
    dump_json(rel2id, "formatted/sent/rel2id.json", indent=2)

    sent_train_1 = load_csv("raw/IPRE/data/train/sent_train_1.txt",
                            title_row=False, title_keys=["id", "head", "tail", "text"])
    sent_train_2 = load_csv("raw/IPRE/data/train/sent_train_2.txt",
                            title_row=False, title_keys=["id", "head", "tail", "text"])
    sent_train = sent_train_1 + sent_train_2
    id2sent = {}
    for ins in sent_train:
        id2sent[ins["id"]] = ins
    sent_train_labels = load_csv("raw/IPRE/data/train/sent_relation_train.txt",
                                 title_row=False, title_keys=["id", "relations"])
    train_data = []
    for ins in sent_train_labels:
        rels = list(map(int, ins["relations"].split()))
        d = id2sent[ins["id"]]
        d["relations"] = list(map(lambda x: id2rel[x], rels))
        train_data.append(d)

    print(len(train_data), train_data[:2])
    dump_line_json(train_data, "formatted/sent/train.line.json")

    bag_relation_train = load_csv("data/IPRE/raw/IPRE/data/train/bag_relation_train.txt",
                                  title_row=False, title_keys=["id", "head", "tail", "sent_ids", "relations"])
