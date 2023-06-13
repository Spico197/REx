import json
import random
from collections import defaultdict
from pathlib import Path

from rex.utils.io import dump_jsonlines, load_jsonlines


def get_instruction_pool(data):
    instruction_pool = set()
    for d in data:
        instruction_pool.add(d["instruction"])
    return list(instruction_pool)


def get_schema(uie_data):
    schema = {"ent": set(), "rel": set(), "event": defaultdict(set)}
    for d in uie_data:
        for ent in d["entity"]:
            schema["ent"].add(ent["type"])
        for rel in d["relation"]:
            schema["rel"].add(rel["type"])
        for event in d["event"]:
            schema["event"][event["type"]].update(
                {arg["type"] for arg in event["args"]}
            )
    schema["ent"] = list(schema["ent"])
    schema["rel"] = list(schema["rel"])
    schema["event"] = {k: list(v) for k, v in schema["event"].items()}
    return schema


def convert_uie_to_udi(ins, schema, instruction, idx):
    def _get_span(tokens, offset, text):
        # +1: add one space between tokens
        s_off = len(" ".join(tokens[: offset[0]])) + 1
        e_off = s_off + len(text)
        return [s_off, e_off]

    udi_ins = {
        "id": idx,
        "instruction": instruction,
        "schema": schema,
        "ans": {
            "ent": [],
            "rel": [],
            "event": [],
        },
        "text": " ".join(ins["tokens"]),
        "bg": "",
    }
    for ent in ins["entity"]:
        _e = {
            "type": ent["type"],
            "text": ent["text"],
            "span": _get_span(ins["tokens"], ent["offset"], ent["text"]),
        }
        udi_ins["ans"]["ent"].append(_e)
    for rel in ins["relation"]:
        _r = {
            "relation": rel["type"],
            "head": {
                "text": rel["args"][0]["text"],
                "span": _get_span(
                    ins["tokens"], rel["args"][0]["offset"], rel["args"][0]["text"]
                ),
            },
            "tail": {
                "text": rel["args"][1]["text"],
                "span": _get_span(
                    ins["tokens"], rel["args"][1]["offset"], rel["args"][1]["text"]
                ),
            },
        }
        udi_ins["ans"]["rel"].append(_r)
    for event in ins["event"]:
        _e = {
            "event_type": event["type"],
            "trigger": {
                "text": event["text"],
                "span": _get_span(ins["tokens"], event["offset"], event["text"]),
            },
            "args": [],
        }
        for arg in event["args"]:
            _a = {
                "role": arg["type"],
                "text": arg["text"],
                "span": _get_span(ins["tokens"], arg["offset"], arg["text"]),
            }
            _e["args"].append(_a)
        udi_ins["ans"]["event"].append(_e)
    return udi_ins


def convert_data(
    id_prefix,
    uie_input_filepath,
    udi_input_filepath,
    output_filepath,
    schema,
    instructions,
):
    uie_data = load_jsonlines(uie_input_filepath)
    udi_data = load_jsonlines(udi_input_filepath)
    converted_udi_data = []
    for i, d in enumerate(uie_data):
        udi_data = convert_uie_to_udi(
            d, schema, random.choice(instructions), f"{id_prefix}.{i}"
        )
        converted_udi_data.append(udi_data)
    out_p = Path(output_filepath)
    if not out_p.parent.exists():
        out_p.parent.mkdir(parents=True)

    dump_jsonlines(converted_udi_data, out_p)


def easy_convert(uie_data_dir, udi_data_dir, output_dir):
    uie_data_dir = Path(uie_data_dir)
    udi_data_dir = Path(udi_data_dir)
    output_dir = Path(output_dir)

    task_type = uie_data_dir.parent.stem
    data_name = uie_data_dir.stem

    train_path = uie_data_dir / "train.json"
    dev_path = uie_data_dir / "val.json"
    test_path = uie_data_dir / "test.json"
    train_data = load_jsonlines(train_path)
    dev_data = load_jsonlines(dev_path)
    test_data = load_jsonlines(test_path)
    tot_uie_data = train_data + dev_data + test_data
    schema = get_schema(tot_uie_data)

    train_path = udi_data_dir / "train.jsonl"
    dev_path = udi_data_dir / "dev.jsonl"
    test_path = udi_data_dir / "test.jsonl"
    train_data = load_jsonlines(train_path)
    dev_data = load_jsonlines(dev_path)
    test_data = load_jsonlines(test_path)
    tot_udi_data = train_data + dev_data + test_data
    instructions = get_instruction_pool(tot_udi_data)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for uie_split in ["train", "val", "test"]:
        udi_split = "dev" if uie_split == "val" else uie_split
        convert_data(
            f"{task_type}.{data_name}.{udi_split}",
            uie_data_dir / f"{uie_split}.json",
            udi_data_dir / f"{udi_split}.jsonl",
            output_dir / f"{udi_split}.jsonl",
            schema,
            instructions,
        )


def main():
    # absa
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/absa/14lap",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/14lap/instructed",
        "/data/tzhu/UIE/outputs_data/absa/14lap",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/absa/14res",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/14res/instructed",
        "/data/tzhu/UIE/outputs_data/absa/14res",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/absa/15res",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/15res/instructed",
        "/data/tzhu/UIE/outputs_data/absa/15res",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/absa/16res",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/16res/instructed",
        "/data/tzhu/UIE/outputs_data/absa/16res",
    )
    # ent
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/entity/conll03",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/ent/en/CoNLL2003/instructed",
        "/data/tzhu/UIE/outputs_data/ent/conll03",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/entity/mrc_ace04",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/ent/en/ACE_2004/instructed",
        "/data/tzhu/UIE/outputs_data/ent/ace04",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/entity/mrc_ace05",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/ent/en/ACE05-EN-plus/instructed",
        "/data/tzhu/UIE/outputs_data/ent/ace05",
    )
    # event
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/event/casie",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/event/en/CASIE/instructed",
        "/data/tzhu/UIE/outputs_data/event/casie",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/event/oneie_ace05_en_event",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/event/en/ACE05-EN-plus/fixed_instructed",
        "/data/tzhu/UIE/outputs_data/event/ace05-evt",
    )
    # rel
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/relation/ace05-rel",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/ACE05-EN-plus/instructed",
        "/data/tzhu/UIE/outputs_data/rel/ace05-rel",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/CoNLL2004/instructed",
        "/data/tzhu/UIE/outputs_data/rel/conll04",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/relation/NYT",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/NYT_multi/instructed",
        "/data/tzhu/UIE/outputs_data/rel/nyt",
    )
    easy_convert(
        "/data/tzhu/UIE/dataset_processing/converted_data/text2spotasoc/relation/scierc",
        "/data/tzhu/Mirror/resources/Mirror/v1.3/rel/en/sciERC/instructed",
        "/data/tzhu/UIE/outputs_data/rel/scierc",
    )


if __name__ == "__main__":
    main()
