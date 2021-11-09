from typing import Type, Union, List, Any


def find_all_positions(
    long: Union[List[Any], str], sub: Union[List[Any], str]
) -> List[List[int]]:
    """Find all sub list positions in the long list

    Args:
        long: list or string to positions in
        sub: short list or string to find

    Returns:
        all the appeared positions

    Raises:
        ValueError: if the len of `sub` is longer than the `long`
        ValueError: if the two input types are not the same

    Examples:
        >>> long_str = "123123123"
        >>> short_str = "123"
        >>> find_all_positions(long_str, short_str)
        [[0, 3], [3, 6], [6, 9]]
        >>> long_list = ["123", "1234", "12345"]
        >>> short_list = ["123"]
        >>> find_all_positions(long_list, short_list)
        [[0, 1]]
    """
    if isinstance(long, str) and isinstance(sub, str):
        long = list(long)
        sub = list(sub)
    if isinstance(long, list) and isinstance(sub, list):
        if len(sub) > len(long):
            raise ValueError("sub length is longer than the long")
        positions = []
        len_sub = len(sub)
        for idx in range(0, len(long) - len(sub) + 1):
            if sub == long[idx : idx + len_sub]:
                positions.append([idx, idx + len_sub])
        return positions
    else:
        raise TypeError("types of the two input must be str or list")


def construct_relative_positions(pos: int, max_length: int) -> List[int]:
    """Construct relative positions to a specified pos

    Args:
        pos: the pos that will be `0`
        max_length: max sequence length

    Returns:
        a list of relative positions

    Raises:
        ValueError: if pos is less than 0 or greater equal than max_length
    """
    if pos < 0 or pos >= max_length:
        raise ValueError(f"pos: {pos} is not in [0, {max_length})")
    positions = list(range(0, max_length, 1))
    positions = list(map(lambda x: abs(x - pos), positions))
    return positions
