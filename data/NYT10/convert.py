import os

from rex.utils.io import load_line_json_iterator, dump_line_json
from rex.utils.position import find_all_positions


def convert_data(dataset_name, filepath):
    final_data = []
    for idx, ins in enumerate(load_line_json_iterator(filepath)):
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
    dump_line_json(final_data, f"formatted/{dataset_name}.linejson")


if __name__ == "__main__":
    os.makedirs("formatted", exist_ok=True)
    for dn in ["train", "test"]:
        convert_data(dn, f"raw/nyt10_{dn}.txt")
