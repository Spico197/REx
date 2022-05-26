from collections import defaultdict

from rex.metrics import (
    DEFAULT_PRF1_RESULT_DICT,
    calc_p_r_f1_from_tp_fp_fn,
    safe_division,
)


def get_measures_from_sets(gold_set: set, pred_set: set) -> dict:
    intersection = gold_set & pred_set
    tp = len(intersection)
    fp = len(pred_set - intersection)
    fn = len(gold_set - intersection)
    return calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)


def tagging_prf1(gold_ents, pred_ents, type_idx=1) -> dict:
    """Get tagging P, R and F1 results

    Args:
        gold_ents: list of gold entities for each instance.
            Each entity **must** follow the ``(name, type, pos)`` format.
            e.g. [[(Alan, PER, (0, 4)), (Jack, PER, (5, 9))], [...], ...]
        pred_ents: ditto
        type_idx: index to entity type in entity tuples.
            e.g. `type_idx` in `(Alan, PER, (0, 4))` is `1`

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
    for one_ins_gold_ents, one_ins_pred_ents in zip(gold_ents, pred_ents):
        _result = get_measures_from_sets(set(one_ins_gold_ents), set(one_ins_pred_ents))
        measure_results["micro"]["tp"] += _result["tp"]
        measure_results["micro"]["fp"] += _result["fp"]
        measure_results["micro"]["fn"] += _result["fn"]

        _type2ent = defaultdict(lambda: {"gold": set(), "pred": set()})
        for ent in one_ins_gold_ents:
            ent_type = ent[type_idx]
            _type2ent[ent_type]["gold"].add(ent)
        for ent in one_ins_pred_ents:
            ent_type = ent[type_idx]
            _type2ent[ent_type]["pred"].add(ent)
        for ent_type in _type2ent:
            _result = get_measures_from_sets(
                _type2ent[ent_type]["gold"], _type2ent[ent_type]["pred"]
            )
            measure_results["types"][ent_type]["tp"] += _result["tp"]
            measure_results["types"][ent_type]["fp"] += _result["fp"]
            measure_results["types"][ent_type]["fn"] += _result["fn"]

    # micro
    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(
        measure_results["micro"]["tp"],
        measure_results["micro"]["fp"],
        measure_results["micro"]["fn"],
    )

    # for each type
    for ent_type in measure_results["types"]:
        measure_results["types"][ent_type] = calc_p_r_f1_from_tp_fp_fn(
            measure_results["types"][ent_type]["tp"],
            measure_results["types"][ent_type]["fp"],
            measure_results["types"][ent_type]["fn"],
        )

    measure_results["types"] = dict(measure_results["types"])

    # macro
    macro_results = defaultdict(list)
    for ent_type in measure_results["types"]:
        for key in measure_results["types"][ent_type]:
            macro_results[key].append(measure_results["types"][ent_type][key])
    for key in macro_results:
        measure_results["macro"][key] = safe_division(
            sum(macro_results[key]), len(macro_results[key])
        )

    return measure_results
