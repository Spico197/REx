from rex.utils.progress_bar import tqdm


def test_pbar():
    pbar = tqdm(range(10))
    for _ in pbar:
        pass
    pbar_str = str(pbar)

    assert "#" in pbar_str
    assert "100%" in pbar_str
