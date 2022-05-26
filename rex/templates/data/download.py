import os

from datasets import load_dataset

from rex.data.label_encoder import LabelEncoder
from rex.utils.build_emb import build_emb
from rex.utils.io import dump_iterable, dump_jsonlines

cache_dir = "./data/cache"
dump_dir = "./data/formatted"
# downloaded emb file from https://github.com/Embedding/Chinese-Word-Vectors
input_embedding_filepath = (
    "/data4/tzhu/CCKS2020-FinEBody-nonNaN/data/sgns.sogounews.bigram-char"
)
output_embedding_filepath = "./data/formatted/vocab.emb"

datasets = load_dataset("peoples_daily_ner", cache_dir=cache_dir)
lbe = LabelEncoder()

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def extract(dataset):
    to_tag = dataset.info.features["ner_tags"].feature.int2str
    data = []
    tokens = set()
    for d in dataset:
        if len(d["tokens"]) < 1:
            continue
        tokens.update(d["tokens"])
        d["ner_tags"] = list(map(to_tag, d["ner_tags"]))
        lbe.update(d["ner_tags"])
        data.append(d)
    return data, tokens


vocab_files = []
for key in datasets:
    data, tokens = extract(datasets[key])
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))
    vocab_filepath = os.path.join(cache_dir, f"{key}.vocab")
    vocab_files.append(vocab_filepath)
    dump_iterable(tokens, vocab_filepath)

lbe.save_pretrained(os.path.join(dump_dir, "label2id.json"))
build_emb(input_embedding_filepath, output_embedding_filepath, *vocab_files)
