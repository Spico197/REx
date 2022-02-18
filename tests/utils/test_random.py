from rex.utils.random import generate_random_string


def test_random():
    rand_str = generate_random_string(4)
    assert len(rand_str) == 4
    for c in rand_str:
        assert "a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9"
