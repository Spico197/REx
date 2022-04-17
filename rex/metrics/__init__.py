DEFAULT_PRF1_RESULT_DICT = {"p": 0.0, "r": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def safe_division(numerator, denominator):
    try:
        val = numerator / denominator
    except ZeroDivisionError:
        val = 0.0
    return val


def calc_p_r_f1_from_tp_fp_fn(tp, fp, fn):
    p = safe_division(tp, tp + fp)
    r = safe_division(tp, tp + fn)
    f1 = safe_division(2 * p * r, p + r)

    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
