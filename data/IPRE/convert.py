import os
import json

from rex.io import utils as ru
from rex.io.transform import position as rtp


def convert_data(sentence_filepath, 
                 sent_relation_filepath,
                 dataset_name):
    sents = ru.load_csv(sentence_filepath,
                     title_row=False,
                     title_keys=["id", "head", "tail", "text"])
    id2sent = {}
    for ins in sents:
        id2sent[ins["id"]] = ins
    sent_labels = ru.load_csv(sent_relation_filepath,
                           title_row=False, title_keys=["id", "relations"])
    final_data = []
    for ins in sent_labels:
        rels = list(set(map(int, ins["relations"].split())))
        d = id2sent[ins["id"]]
        d["relations"] = list(map(lambda x: id2rel[x], rels))
        for word in ["head", "tail"]:
            d[word] = d[word].replace(" ", "")
            d["text"] = d["text"].replace(" ", "")
            positions = rtp.find_all_positions(d["text"], d[word])
            if len(positions) == 0:
                print("warn! data skipped", d)
                continue
            d[word] = {
                "word": d[word],
                "positions": positions
            }
        final_data.append(d)

    print(len(final_data), final_data[:2])
    ru.dump_line_json(final_data, f"formatted/{dataset_name}.line.json")


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)
    rel2ids = ru.load_csv("raw/IPRE/data/relation2id.txt",
                       title_row=False, title_keys=["relation", "id"])
    rel2id = {}
    for ins in rel2ids:
        rel2id[ins["relation"]] = int(ins["id"])
    id2rel = {id_: rel for rel, id_ in rel2id.items()}
    ru.dump_json(rel2id, "formatted/rel2id.json", indent=2)

    for dn in ["train", "dev", "test"]:
        convert_data(f"raw/IPRE/data/{dn}/sent_{dn}.txt",
                     f"raw/IPRE/data/{dn}/sent_relation_{dn}.txt",
                     dn)
