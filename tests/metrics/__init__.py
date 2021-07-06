import unittest

from rex.metrics import calc_p_r_f1_from_tp_fp_fn


class TestMetricInit(unittest.TestCase):
    def test_p_r_f1_calc_from_tp_fp_fn(self):
        tp = fp = fn = 1
        results = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn, eps=0.0)
        self.assertEqual(results['p'], 0.5)
        self.assertEqual(results['r'], 0.5)
        self.assertEqual(results['f1'], 0.5)
