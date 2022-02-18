from collections import defaultdict
from typing import Iterable, Mapping, Optional

import numpy as np
from sklearn import metrics

from rex.metrics import calc_p_r_f1_from_tp_fp_fn


def accuracy(preds, golds):
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    correct = total = 0
    for pred, gold in zip(preds, golds):
        total += 1
        if pred == gold:
            correct += 1
    return correct / total


def mcml_prf1(preds, golds):
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
    preds,
    golds,
    num_classes=-1,
    ignore_labels: Optional[Iterable] = None,
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
    measure_results = defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0})

    if num_classes > 0:
        # specified number of classes
        labels = list(range(num_classes))
    else:
        # guess the number of classes through current input
        labels = None

    MCM = metrics.multilabel_confusion_matrix(
        golds, preds, sample_weight=None, labels=labels, samplewise=False
    )

    tp, tp_fp, tp_fn = [], [], []

    if ignore_labels is None:
        ignore_labels = []
    for tmp_label in range(MCM.shape[0]):
        if tmp_label not in ignore_labels:
            tp.append(MCM[tmp_label, 1, 1])
            tp_fp.append(MCM[tmp_label, 1, 1] + MCM[tmp_label, 0, 1])
            tp_fn.append(MCM[tmp_label, 1, 1] + MCM[tmp_label, 1, 0])

    tp = np.stack(tp)
    tp_fp = np.stack(tp_fp)
    tp_fn = np.stack(tp_fn)

    # micro-averaged scores
    tp_sum = tp.sum()
    pred_sum = tp_fp.sum()
    true_sum = tp_fn.sum()
    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(
        tp_sum, pred_sum - tp_sum, true_sum - tp_sum
    )

    # macro-averaged scores
    results = calc_p_r_f1_from_tp_fp_fn(tp, tp_fp - tp, tp_fn - tp)

    measure_results["macro"]["p"] = results["p"].mean()
    measure_results["macro"]["r"] = results["r"].mean()
    measure_results["macro"]["f1"] = results["f1"].mean()

    if label_idx2name is None:
        label_idx2name = {}
    for idx, p, r, f1 in zip(
        range(MCM.shape[0]), results["p"], results["r"], results["f1"]
    ):
        if idx in label_idx2name:
            idx_name = label_idx2name.get(idx)
        else:
            idx_name = idx
        measure_results[idx_name] = {"p": p, "r": r, "f1": f1}

    return dict(measure_results)
