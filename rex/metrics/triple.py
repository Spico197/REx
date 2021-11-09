from collections import defaultdict

from rex.metrics import calc_p_r_f1_from_tp_fp_fn


def measure_triple(preds, golds, eps=1e-12):
    result = {
        "triple": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "subject": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "object": {"p": 0.0, "r": 0.0, "f1": 0.0},
        "relation": {"p": 0.0, "r": 0.0, "f1": 0.0},
    }
    middle_stat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for pred, gold in zip(preds, golds):
        pred_triples = set(pred)
        gold_triples = set(gold)
        middle_stat["triple"]["tp"] += len(pred_triples & gold_triples)
        middle_stat["triple"]["fp"] += len(pred_triples - gold_triples)
        middle_stat["triple"]["fn"] += len(gold_triples - pred_triples)

        pred_relations = set(x[1] for x in pred_triples)
        gold_relations = set(x[1] for x in gold_triples)
        middle_stat["relation"]["tp"] += len(pred_relations & gold_relations)
        middle_stat["relation"]["fp"] += len(pred_relations - gold_relations)
        middle_stat["relation"]["fn"] += len(gold_relations - pred_relations)

        pred_subjs = set(x[0] for x in pred_triples)
        gold_subjs = set(x[0] for x in gold_triples)
        middle_stat["subject"]["tp"] += len(pred_subjs & gold_subjs)
        middle_stat["subject"]["fp"] += len(pred_subjs - gold_subjs)
        middle_stat["subject"]["fn"] += len(gold_subjs - pred_subjs)

        pred_objs = set(x[2] for x in pred_triples)
        gold_objs = set(x[2] for x in gold_triples)
        middle_stat["object"]["tp"] += len(pred_objs & gold_objs)
        middle_stat["object"]["fp"] += len(pred_objs - gold_objs)
        middle_stat["object"]["fn"] += len(gold_objs - pred_objs)

    for measure in result:
        result[measure] = calc_p_r_f1_from_tp_fp_fn(**middle_stat[measure], eps=eps)
        result[measure].update(middle_stat[measure])

    return result
