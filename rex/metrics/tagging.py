from collections import defaultdict
from functools import reduce

from rex.metrics import calc_p_r_f1_from_tp_fp_fn


def tagging_prf1(gold_ents, pred_ents):
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

    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(
        global_tp, global_fp, global_fn
    )
    return measure_results
