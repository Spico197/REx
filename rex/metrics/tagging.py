from collections import defaultdict
from functools import reduce


def tagging_prf1(gold_ents, pred_ents, eps=1e-12):
    # TODO: add macro and type-specified support
    measure_results = defaultdict(lambda: {"p": 0.0, "r": 0.0, "f1": 0.0})
    global_tp = global_fp = global_fn = 0
    for gold_ents, pred_ents in zip(gold_ents, pred_ents):
        gold_ents = set(reduce(lambda x, y: x + y, gold_ents.values(), []))
        pred_ents = set(reduce(lambda x, y: x + y, pred_ents.values(), []))
        intersection = gold_ents & pred_ents
        global_tp += len(intersection)
        global_fp += len(pred_ents - intersection)
        global_fn += len(gold_ents - intersection)

    p = measure_results["micro"]["p"] = global_tp / (global_tp + global_fp + eps)
    r = measure_results["micro"]["r"] = global_tp / (global_tp + global_fn + eps)
    measure_results["micro"]["f1"] = 2 * p * r / (p + r + eps)
    return measure_results
