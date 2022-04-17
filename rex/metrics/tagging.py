from collections import defaultdict

from rex.metrics import (
    DEFAULT_PRF1_RESULT_DICT,
    safe_division,
    calc_p_r_f1_from_tp_fp_fn,
)


def get_measures_from_sets(gold_set: set, pred_set: set) -> dict:
    intersection = gold_set & pred_set
    tp = len(intersection)
    fp = len(pred_set - intersection)
    fn = len(gold_set - intersection)
    return calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)


def tagging_prf1(gold_ents, pred_ents):
    """Get tagging P, R and F1 results

    Args:
        gold_ents: list of batch of gold entities.
            Each entity **must** follow the ``(name, type, pos)`` format.
        pred_ents: ditto

    Returns:
        {
            "micro": {"p": 0.0, ...},
            "macro": {"p": 0.0, ...},
            "types": {
                "type1": {"p": 0.0, ...}
            }
        }
    """
    measure_results = {
        "micro": DEFAULT_PRF1_RESULT_DICT.copy(),
        "macro": DEFAULT_PRF1_RESULT_DICT.copy(),
        "types": defaultdict(lambda: DEFAULT_PRF1_RESULT_DICT.copy()),
    }
    ents = {
        "all": {"gold": set(), "pred": set()},
        "types": defaultdict(lambda: {"gold": set(), "pred": set()}),
    }
    for batch_gold_ents, batch_pred_ents in zip(gold_ents, pred_ents):
        ents["all"]["gold"].update(batch_gold_ents)
        ents["all"]["pred"].update(batch_pred_ents)

        for gold_ent in batch_gold_ents:
            ent_type = gold_ent[1]
            ents["types"][ent_type]["gold"].add(gold_ent)
        for pred_ent in batch_pred_ents:
            ent_type = pred_ent[1]
            ents["types"][ent_type]["pred"].add(pred_ent)

    # micro
    measure_results["micro"] = get_measures_from_sets(
        ents["all"]["gold"], ents["all"]["pred"]
    )

    # for each type
    for ent_type in ents["types"]:
        measure_results["types"][ent_type] = get_measures_from_sets(
            ents["types"][ent_type]["gold"],
            ents["types"][ent_type]["pred"],
        )

    # macro
    macro_results = defaultdict(lambda: defaultdict(list))
    for ent_type in measure_results["types"]:
        for key in measure_results["types"][ent_type]:
            macro_results[key].append(measure_results["types"][ent_type][key])
    for key in measure_results["types"][ent_type]:
        measure_results["macro"][key] = safe_division(
            sum(macro_results[key]), len(macro_results[key])
        )

    return measure_results
