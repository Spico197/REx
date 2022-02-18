def calc_p_r_f1_from_tp_fp_fn(tp, fp, fn):
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0.0

    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0.0

    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0.0

    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
