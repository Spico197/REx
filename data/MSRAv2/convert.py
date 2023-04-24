from pathlib import Path

from rex.utils.io import dump_json, dump_jsonlines, load_csv

if __name__ == "__main__":
    type2query = {
        "NR": "人名和虚构的人物形象",
        "NS": "按照地理位置划分的国家,城市,乡镇,大洲",
        "NT": "组织包括公司,政府党派,学校,政府,新闻机构",
    }

    data_dir = Path("data/MSRAv2/raw")
    out_dir = Path("data/MSRAv2/formatted")
    if not out_dir.exists():
        out_dir.mkdir()
    dump_json(type2query, out_dir / "role2query.json")

    for dname in ["train.char.bmes", "dev.char.bmes", "test.char.bmes"]:
        data = load_csv(data_dir / dname, False, sep=" ")
        new_data = []
        tokens = []
        ner_tags = []
        idx = 0
        for line in data:
            if len(line) == 2:
                tokens.append(line[0])
                ner_tags.append(line[1])
            else:
                if len(tokens) < 1:
                    continue
                new_data.append(
                    {
                        "id": f"{dname}.{idx}",
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                )
                idx += 1

                tokens = []
                ner_tags = []
        dump_jsonlines(new_data, out_dir / f"{dname.removesuffix('.char.bmes')}.jsonl")
