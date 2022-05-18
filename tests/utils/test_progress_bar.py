from rex.utils.progress_bar import rbar


def test_pbar():
    pbar = rbar(10)
    for _ in pbar:
        pass
    pbar_str = str(pbar)

    assert "#" in pbar_str
    assert "100%" in pbar_str
