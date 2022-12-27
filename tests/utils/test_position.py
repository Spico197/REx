import pytest

from rex.utils import position


def test_find_all_positions():
    assert position.find_all_positions("123123123", "123") == [(0, 3), (3, 6), (6, 9)]


def test_construct_relative_positions():
    case = (2, 5)
    result = [2, 1, 0, 1, 2]
    assert position.construct_relative_positions(*case) == result


def test_longer_sub_string():
    long = list("123456")
    short = list("1234567")
    with pytest.raises(ValueError):
        position.find_all_positions(long, short)


def test_type_check():
    long = "123456"
    short = 1234
    with pytest.raises(TypeError):
        position.find_all_positions(long, short)


def test_relative_position_construction_err():
    with pytest.raises(ValueError):
        position.construct_relative_positions(81, 80)
