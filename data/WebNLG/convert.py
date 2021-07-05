import os
from collections import Counter

import numpy as np

from rex.utils.io import (
    dump_csv,
    dump_iterable,
    dump_json,
    load_json,
    dump_line_json
)


def convert_data(dataset_name, filepath):
    data = load_json(filepath)
    sents, spos = data
    final_data = []
    lens = []
    for sent_idx, sent in enumerate(sents):
        tokens = list(map(lambda x: id2word[x], sent))
        lens.append(len(tokens))
        d = {
            "id": f"{dataset_name.upper()}.{sent_idx}",
            "tokens": tokens,  # char tokenize
            "entities": [],
            "relations": []
        }
        triples = np.reshape(spos[sent_idx], (-1, 3)).tolist()
        for triple in triples:
            head_ent = ["ENTITY", triple[0], triple[0] + 1, tokens[triple[0]]]
            if head_ent not in d['entities']:
                d['entities'].append(head_ent)
            tail_ent = ["ENTITY", triple[1], triple[1] + 1, tokens[triple[1]]]
            if tail_ent not in d['entities']:
                d['entities'].append(tail_ent)
            relation = id2rel[triple[2]]
            d['relations'].append([
                relation, d['entities'].index(head_ent), d['entities'].index(tail_ent),
                [head_ent[3], tail_ent[3]]
            ])
        final_data.append(d)

    print(f'len of {dataset_name}:', len(final_data), final_data[:2])
    dump_line_json(final_data, f"formatted/{dataset_name}.linejson")

    len_counter = Counter(lens)
    print(max(lens), len_counter.most_common())


def convert_word_vec(filepath):
    word_id2vec = load_json(filepath)
    tot_num = len(word_id2vec)
    emb_len = len(list(word_id2vec.values())[0])
    word_vec = [[str(tot_num), str(emb_len)]]
    word_vec = word_vec + [[id2word[int(word_idx)], *list(map(str, word_emb))] for word_idx, word_emb in word_id2vec.items()]
    dump_csv(word_vec, "formatted/word.vec", delimiter=' ')
    dump_iterable(word2id.keys(), "formatted/vocab.txt")


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)
    rel2id = load_json("raw/relations2id.json")
    dump_json(rel2id, "formatted/rel2id.json")

    id2rel = {id_: rel for rel, id_ in rel2id.items()}
    word2id = load_json("raw/words2id.json")
    id2word = {id_: word for word, id_ in word2id.items()}

    for dn in ["train", "dev", "valid"]:
        convert_data(dn, f"raw/{dn}.json")

    convert_word_vec("raw/words_id2vector.json")
