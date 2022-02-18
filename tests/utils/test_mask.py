import pytest

from rex.utils.mask import construct_piecewise_mask


def test_piecewise_mask():
    mask = construct_piecewise_mask(2, 7, 10, 15)
    assert mask == [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0]
    mask = construct_piecewise_mask(0, 7, 10, 15)
    assert mask == [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0]
    mask = construct_piecewise_mask(7, 1, 10, 15)
    assert mask == [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0]

    with pytest.raises(AssertionError):
        construct_piecewise_mask(-1, 0, 0, 0)

    with pytest.raises(AssertionError):
        construct_piecewise_mask(0, -1, 0, 0)
