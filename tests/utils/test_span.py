import unittest

import torch

from rex.utils import span


class TestSpan(unittest.TestCase):
    def test_find_closest_span_pairs_common_case(self):
        head = torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
        tail = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.long)
        self.assertEqual(
            span.find_closest_span_pairs(head, tail, backtrace=False),
            [(0, 1), (3, 3), (6, 6)])

    def test_find_closest_span_pairs_backtrace(self):
        head = torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
        tail = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.long)
        self.assertEqual(
            span.find_closest_span_pairs(head, tail, backtrace=True),
            [(0, 1), (3, 3), (6, 6), (3, 5)])

    def test_find_closest_span_pairs_backtrace_mh_st(self):
        head = torch.tensor([1, 0, 1, 1, 0, 0, 1], dtype=torch.long)
        tail = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.long)
        self.assertEqual(
            span.find_closest_span_pairs(head, tail, backtrace=True),
            [(0, 1), (2, 3), (3, 3), (6, 6), (3, 5)])

    def test_find_closest_span_pairs_with_index(self):
        head = torch.tensor([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]], dtype=torch.long)
        tail = torch.tensor([[0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0]], dtype=torch.long)
        self.assertEqual(
            span.find_closest_span_pairs_with_index(head, tail, backtrace=False),
            [(0, 0, 1), (0, 3, 3), (0, 6, 6), (1, 0, 1), (1, 3, 5)])

    def test_find_closest_span_pairs_with_index_backtrace(self):
        head = torch.tensor([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]], dtype=torch.long)
        tail = torch.tensor([[0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0]], dtype=torch.long)
        self.assertEqual(
            span.find_closest_span_pairs_with_index(head, tail, backtrace=True),
            [(0, 0, 1), (0, 3, 3), (0, 6, 6), (0, 3, 5), (1, 0, 1), (1, 3, 5)])
