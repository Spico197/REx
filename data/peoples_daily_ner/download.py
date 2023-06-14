import os

# pip install datasets
from datasets import load_dataset

from rex.data.label_encoder import LabelEncoder
from rex.utils.io import dump_iterable, dump_json, dump_jsonlines

cache_dir = "./data/cache"
dump_dir = "./data/formatted"

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
    if key == "validation":
        key = "dev"
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))
    vocab_filepath = os.path.join(cache_dir, f"{key}.vocab")
    vocab_files.append(vocab_filepath)
    dump_iterable(tokens, vocab_filepath)

lbe.save_pretrained(os.path.join(dump_dir, "label2id.json"))

# https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/fix%2Fyuxian/ner2mrc/queries/zh_msra.json
dump_json(
    {
        "LOC": "按照地理位置划分的国家,城市,乡镇,大洲",
        "PER": "人名和虚构的人物形象",
        "ORG": "组织包括公司,政府党派,学校,政府,新闻机构",
    },
    os.path.join(dump_dir, "role2query.json"),
)
