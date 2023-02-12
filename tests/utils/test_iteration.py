from rex.utils import iteration


def test_flatten_all():
    nested = [1, 3, [3, 4, [5, 6]], [7, 8]]
    res = []
    for el in iteration.flatten_all_iter(nested):
        res.append(el)
    assert res == [1, 3, 3, 4, 5, 6, 7, 8]


def test_windowed_queue():
    queue = [1, 2, 3, 4, 5]
    res = []
    for elems in iteration.windowed_queue_iter(queue, 2):
        res.append(elems)
    assert res == [[1, 2], [3, 4], [5]]
