import unittest

from rex.io.transform import position

class TestPosition(unittest.TestCase):
    def test_find_all_positions(self):
        self.assertEqual(
            position.find_all_positions("123123123", "123"),
            [[0, 3], [3, 6], [6, 9]])
