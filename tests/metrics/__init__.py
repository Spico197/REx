from rex.metrics import calc_p_r_f1_from_tp_fp_fn


def test_p_r_f1_calc_from_tp_fp_fn():
    tp = fp = fn = 1
    results = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn, eps=0.0)
    assert results["p"] == 0.5
    assert results["r"] == 0.5
    assert results["f1"] == 0.5
