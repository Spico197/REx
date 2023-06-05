import os
import tempfile

from rex.utils.io import find_files


def test_find_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "a"), exist_ok=True)
        for i in range(2):
            with open(os.path.join(tmpdir, f"{i}.txt"), "w") as f:
                f.write(str(i))
        with open(os.path.join(tmpdir, "a", "sub.txt"), "w") as f:
            f.write(str(i))

        found_files = find_files(r".*\.txt", tmpdir, recursive=False)
        assert set(found_files) == {os.path.join(tmpdir, f"{i}.txt") for i in range(2)}

        found_recursive_files = find_files(r".*\.txt", tmpdir)
        assert set(found_recursive_files) == {*found_files, os.path.join(tmpdir, "a", "sub.txt")}
