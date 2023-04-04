import json
import zipfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from rex.utils.io import load_jsonlines
from rex.utils.progress_bar import pbar

BASE_DIR = "data/EncyclopediaDS/long"


def main():
    filepath = f"{BASE_DIR}/result.zip"
    output = open(f"{BASE_DIR}/pred.jsonl", "a", encoding="utf8")
    output_99 = open(f"{BASE_DIR}/pred.99.jsonl", "a", encoding="utf8")
    output_98 = open(f"{BASE_DIR}/pred.98.jsonl", "a", encoding="utf8")
    output_97 = open(f"{BASE_DIR}/pred.97.jsonl", "a", encoding="utf8")

    probs = []
    label_to_num = defaultdict(lambda: 0)
    with zipfile.ZipFile(filepath, "r") as zip_file:
        filename = zip_file.namelist()[0]
        with zip_file.open(filename, "r") as fin:
            bar = pbar(fin)
            for line in bar:
                ins = json.loads(line)
                ins_string = f"{json.dumps(ins, ensure_ascii=False)}"
                prob = ins["prob"][1]
                probs.append(prob)
                label_to_num[ins["predicted_label"]] += 1
                if ins["predicted_label"] == 1:
                    output.write(f"{ins_string}\n")
                if prob > 0.99:
                    output_99.write(f"{ins_string}\n")
                if prob > 0.98:
                    output_98.write(f"{ins_string}\n")
                if prob > 0.97:
                    output_97.write(f"{ins_string}\n")

                bar.set_postfix_str(str(label_to_num))
    print(dict(label_to_num))

    arr = np.array(probs)
    np.save(f"{BASE_DIR}/probs.npy", arr)
    plt.hist(arr, bins=500)
    plt.savefig("probs.png")

    output.close()
    output_99.close()
    output_98.close()
    output_97.close()


def filter_by_prob(input_filepath, output_filepath, prob=0.999):
    data = load_jsonlines(input_filepath)
    new_data = filter(lambda ins: ins["prob"][1] > prob, data)
    fout = open(output_filepath, "w", encoding="utf8")
    for ins in new_data:
        fout.write(f"{json.dumps(ins, ensure_ascii=False)}\n")
    fout.close()


def stat_relations(input_filepath):
    data = load_jsonlines(input_filepath)
    relations = set(ins["triple"][1] for ins in data)
    print(len(relations))


if __name__ == "__main__":
    # main()
    # filter_by_prob(
    #     f"{BASE_DIR}/pred.99.jsonl",
    #     f"{BASE_DIR}/pred.999.jsonl",
    #     prob=0.999,
    # )
    # filter_by_prob(
    #     f"{BASE_DIR}/pred.999.jsonl",
    #     f"{BASE_DIR}/pred.9999.jsonl",
    #     prob=0.9999,
    # )
    # stat_relations(f"{BASE_DIR}/pred.99.jsonl")     # 19322, num: 5537243
    # stat_relations(f"{BASE_DIR}/pred.999.jsonl")    # 5835, num: 3907591
    # stat_relations(f"{BASE_DIR}/pred.9999.jsonl")   # 1408, num: 1074576

    stat_relations("data/EncyclopediaDS/merged/pred.99.jsonl")
