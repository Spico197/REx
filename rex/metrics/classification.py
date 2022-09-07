from collections import defaultdict
from typing import Iterable, List, Mapping, Optional

import numpy as np
from sklearn import metrics

from rex.metrics import DEFAULT_PRF1_RESULT_DICT, calc_p_r_f1_from_tp_fp_fn


def accuracy(preds: List[int], golds: List[int]):
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    correct = total = 0
    for pred, gold in zip(preds, golds):
        total += 1
        if pred == gold:
            correct += 1
    return correct / total


def mcml_prf1(preds: List[int], golds: List[int]):
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    measure_results = defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0})
    supports = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})
    for pred, gold in zip(preds, golds):
        if len(pred) != len(gold):
            raise ValueError(
                f"Pred: {pred} cannot be matched with gold: {gold} in length."
            )
        for type_idx, (p, g) in enumerate(zip(pred, gold)):
            if p == 1 and g == 1:
                supports[type_idx]["tp"] += 1
            elif p == 1 and g == 0:
                supports[type_idx]["fp"] += 1
            elif p == 0 and g == 1:
                supports[type_idx]["fn"] += 1
    g_tp = g_fp = g_fn = 0.0
    ps = []
    rs = []
    f1s = []
    for type_idx in supports:
        measure_results[type_idx] = calc_p_r_f1_from_tp_fp_fn(
            supports[type_idx]["tp"], supports[type_idx]["fp"], supports[type_idx]["fn"]
        )
        ps.append(measure_results[type_idx]["p"])
        rs.append(measure_results[type_idx]["r"])
        f1s.append(measure_results[type_idx]["f1"])
        g_tp += supports[type_idx]["tp"]
        g_fp += supports[type_idx]["fp"]
        g_fn += supports[type_idx]["fn"]

    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(g_tp, g_fp, g_fn)
    measure_results["macro"]["p"] = sum(ps) / len(ps)
    measure_results["macro"]["r"] = sum(rs) / len(rs)
    measure_results["macro"]["f1"] = sum(f1s) / len(f1s)

    return measure_results


def mc_prf1(
    preds: List[int],
    golds: List[int],
    num_classes: int = -1,
    ignore_labels: Optional[Iterable[int]] = None,
    label_idx2name: Optional[Mapping[int, str]] = None,
):
    """
    get multi-class classification metrics

    Args:
        num_classes: guess the current number of classes if < 0
        ignore_labels: labels to ignore when calculating f1 scores,
            especially helpful when calculating non-NA metrics
    """
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")

    measure_results = {
        "micro": DEFAULT_PRF1_RESULT_DICT.copy(),
        "macro": DEFAULT_PRF1_RESULT_DICT.copy(),
        "types": defaultdict(lambda: DEFAULT_PRF1_RESULT_DICT.copy()),
    }

    if num_classes > 0:
        # specified number of classes
        labels = list(range(num_classes))
    else:
        # guess the number of classes through current input
        labels = None

    mcm = metrics.multilabel_confusion_matrix(
        golds, preds, sample_weight=None, labels=labels, samplewise=False
    )

    tp_list, fp_list, fn_list = [], [], []
    if ignore_labels is None:
        ignore_labels = []
    for tmp_label in range(mcm.shape[0]):
        if tmp_label not in ignore_labels:
            tp_list.append(mcm[tmp_label, 1, 1])
            fp_list.append(mcm[tmp_label, 0, 1])
            fn_list.append(mcm[tmp_label, 1, 0])

    tp_list = np.stack(tp_list)
    fp_list = np.stack(fp_list)
    fn_list = np.stack(fn_list)

    # micro-averaged scores
    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(
        tp_list.sum(), fp_list.sum(), fn_list.sum()
    )

    # macro-averaged scores
    results = calc_p_r_f1_from_tp_fp_fn(tp_list, fp_list, fn_list)

    measure_results["macro"]["p"] = results["p"].mean()
    measure_results["macro"]["r"] = results["r"].mean()
    measure_results["macro"]["f1"] = results["f1"].mean()
    measure_results["macro"]["tp"] = tp_list.mean()
    measure_results["macro"]["fp"] = fp_list.mean()
    measure_results["macro"]["fn"] = fn_list.mean()

    if label_idx2name is None:
        label_idx2name = {}
    for idx, p, r, f1, tp, fp, fn in zip(
        range(mcm.shape[0]),
        results["p"],
        results["r"],
        results["f1"],
        tp_list,
        fp_list,
        fn_list,
    ):
        if idx in label_idx2name:
            idx_name = label_idx2name.get(idx)
        else:
            idx_name = idx
        measure_results["types"][idx_name] = {
            "p": p,
            "r": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    measure_results["types"] = dict(measure_results["types"])
    return measure_results
