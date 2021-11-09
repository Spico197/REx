def construct_piecewise_mask(
    head_pos: int, tail_pos: int, seq_len: int, max_seq_len: int
):
    mask = [0] * max_seq_len
    for i in range(0, max_seq_len):
        if 0 <= i < min(head_pos, tail_pos):
            mask[i] = 1
        elif min(head_pos, tail_pos) <= i < max(head_pos, tail_pos):
            mask[i] = 2
        elif max(head_pos, tail_pos) <= i < min(max_seq_len, seq_len):
            mask[i] = 3
        else:
            mask[i] = 0
    return mask
