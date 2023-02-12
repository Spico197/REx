import pytest

from rex.utils.segmentation import sent_seg, split_list_by_element


def test_seg():
    case = "我说：“翠花，上酸菜。”她说：“欸，好嘞。”"
    sents = sent_seg(case)
    assert sents == ["我说：“翠花，上酸菜。”", "她说：“欸，好嘞。”"]
    sents = sent_seg(case, quotation_seg_mode=False)
    assert sents == ["我说：“翠花，上酸菜。", "”她说：“欸，好嘞。", "”"]

    assert sent_seg("") == []
    with pytest.raises(ValueError):
        sent_seg(123)

    case = "123###456"
    sents = sent_seg(case, special_seg_indicators=[("###", "\n")])
    assert sents == ["123", "456"]

    case = "abcdefg;cgcg.wfwf."
    sents = sent_seg(case, lang="en", punctuations=set(";"))
    assert sents == ["abcdefg;", "cgcg.", "wfwf."]


def test_split_list():
    case = [1, 2, 1, 4, 5]
    res = split_list_by_element(case, 1, keep_empty_segments=False)
    assert res == [[2], [4, 5]]
    res = split_list_by_element(case, 1, keep_empty_segments=True)
    assert res == [[], [2], [4, 5]]
