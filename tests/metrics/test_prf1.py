from rex.metrics import calc_p_r_f1_from_tp_fp_fn


def test_p_r_f1_calc_from_tp_fp_fn():
    tp = fp = fn = 1
    results = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    assert results["p"] == 0.5
    assert results["r"] == 0.5
    assert results["f1"] == 0.5


def test_zero():
    tp = fp = fn = 0
    results = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    assert results["p"] == 0.0
    assert results["r"] == 0.0
    assert results["f1"] == 0.0
