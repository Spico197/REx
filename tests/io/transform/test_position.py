import unittest

from rex.utils import position


class TestPosition(unittest.TestCase):
    def test_find_all_positions(self):
        self.assertEqual(
            position.find_all_positions("123123123", "123"),
            [[0, 3], [3, 6], [6, 9]])

    def test_construct_relative_positions(self):
        case = (2, 5)
        result = [2, 1, 0, 1, 2]
        self.assertEqual(position.construct_relative_positions(*case), result)
