import os

# pip install datasets
from datasets import load_dataset

from rex.utils.io import dump_jsonlines, dump_json


cache_dir = "data/msra_ner/cache"
dump_dir = "data/msra_ner/formatted"

dataset = load_dataset("msra_ner", cache_dir=cache_dir)

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)


def extract(dataset):
    to_tag = dataset.info.features["ner_tags"].feature.int2str
    data = []
    for d in dataset:
        _id = d["id"]
        tokens = d["tokens"]
        if len(tokens) < 1:
            continue
        ner_tags = list(map(to_tag, d["ner_tags"]))
        data.append(
            {
                "id": _id,
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
        )
    return data


for key in dataset:
    data = extract(dataset[key])
    dump_jsonlines(data, os.path.join(dump_dir, f"{key}.jsonl"))

dump_json(
    {
        "LOC": "地名地理位置",
        "PER": "人名公民居民老百姓名人明星",
        "ORG": "公司法院企业集团学校医院单位",
    },
    os.path.join(dump_dir, "role2query.json"),
)
