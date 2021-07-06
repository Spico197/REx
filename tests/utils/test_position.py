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

    def test_longer_sub_string(self):
        long = list("123456")
        short = list("1234567")
        self.assertRaises(ValueError, position.find_all_positions, long, short)

    def test_type_check(self):
        long = "123456"
        short = 1234
        self.assertRaises(TypeError, position.find_all_positions, long, short)

    def test_relative_position_construction_err(self):
        self.assertRaises(ValueError, position.construct_relative_positions, 81, 80)
