import json
import pickle
from typing import Any, DefaultDict, Iterable, List, Optional, OrderedDict

import numpy as np
import torch

from rex.utils.deprecation import deprecation_warning
from rex.utils.logging import logger


def tensor_friendly_json_encoding(obj: Any):
    if isinstance(obj, DefaultDict) or isinstance(obj, OrderedDict):
        obj = dict(obj)
    elif isinstance(obj, set):
        obj = list(obj)
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
    elif isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, torch.Tensor):
        if len(obj.shape) == 0:
            # scalar
            obj = obj.item()
        else:
            obj = obj.tolist()
    return obj


def dump_json(obj, filepath, **kwargs):
    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(
            obj,
            fout,
            ensure_ascii=False,
            default=tensor_friendly_json_encoding,
            **kwargs,
        )


def load_json(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin, **kwargs)
    return data


def dump_line_json(obj, filepath, **kwargs):
    deprecation_warning("dump_line_json", "dump_jsonlines")
    return dump_jsonlines(obj, filepath, **kwargs)


def dump_jsonlines(obj, filepath, **kwargs):
    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = json.dumps(
                d, ensure_ascii=False, default=tensor_friendly_json_encoding, **kwargs
            )
            fout.write("{}\n".format(line_d))


def load_line_json(filepath, **kwargs):
    deprecation_warning("load_line_json", "load_jsonlines")
    return load_jsonlines(filepath, **kwargs)


def load_jsonlines(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


def dump_pickle(obj, filepath, **kwargs):
    with open(filepath, "wb") as fout:
        pickle.dump(obj, fout, **kwargs)


def load_pickle(filepath, **kwargs):
    data = None
    with open(filepath, "rb") as fin:
        data = pickle.load(fin, **kwargs)
    return data


def dump_csv(obj: Iterable[Any], filepath: str, delimiter: Optional[str] = "\t"):
    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = delimiter.join(d)
            fout.write("{}\n".format(line_d))


def load_csv(
    filepath: str,
    title_row: bool,
    title_keys: Optional[List[str]] = None,
    sep: Optional[str] = "\t",
) -> List:
    """load csv file

    Args:
        filepath: filepath to load
        title_row: has title in the first row or not?
                   If true, it'll return a list of dict where keys are from
                   the title, otherwise a list of str list.
        title_keys: if not `title_row`, you can set the title keys yourself.
        sep: separation char
    """
    data = list()
    title_keys = title_keys if title_keys else []
    with open(filepath, "rt", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line_data = line.strip().split(sep)
            if title_row and idx == 0:
                title_keys = line_data
                continue
            if title_keys:
                if len(title_keys) != len(line_data):
                    raise RuntimeError(
                        f"len of title keys: {title_keys}"
                        f" does not match the line data in line {idx + 1}"
                        f" in file: {filepath}"
                    )
                ins = {}
                for col, key in zip(line_data, title_keys):
                    ins[key] = col
            else:
                ins = line_data
            data.append(ins)
    return data


def load_embedding_file(filepath, encoding="utf-8", open_func=open, verbose=False):
    tokens = []
    token2vec = {}
    num_tokens = -1
    dim_emb = 0
    with open_func(filepath, "rt", encoding=encoding) as fin:
        for line_no, line in enumerate(fin):
            line = line.split()
            if line_no == 0:
                if len(line) == 2 and all(x.isdigit() for x in line):
                    num_tokens = int(line[0])
                    dim_emb = int(line[1])
                else:
                    dim_emb = len(line) - 1
                    tokens.append(line[0])
                    token2vec[line[0]] = list(map(float, line[1:]))
                continue
            # dimension checking
            if len(line) - 1 != dim_emb:
                continue
            tokens.append(line[0])
            token2vec[line[0]] = list(map(float, line[1:]))

    if num_tokens > 0 and num_tokens != len(tokens):
        logger.warning(
            f"emb file info num of tokens: {num_tokens}, while {len(tokens)} tokens are found"
        )

    if verbose:
        logger.info(f"Loading #Tokens: {len(tokens)}, Emb dim: {dim_emb}")

    return tokens, token2vec


def load_line_iterator(filepath):
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            yield line


def load_line_json_iterator(filepath, **kwargs):
    deprecation_warning("load_line_json_iterator", "load_jsonlines_iterator")
    return load_jsonlines_iterator(filepath, **kwargs)


def load_jsonlines_iterator(filepath):
    for line in load_line_iterator(filepath):
        yield json.loads(line)


def dump_iterable(obj: Iterable, filepath: str):
    with open(filepath, "wt", encoding="utf-8") as fout:
        for line in obj:
            fout.write(f"{line}\n")
