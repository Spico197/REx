def calc_p_r_f1_from_tp_fp_fn(tp, fp, fn, eps=1e-12):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return {"p": p, "r": r, "f1": f1}
