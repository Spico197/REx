import random
import re
from typing import Any, List, MutableSet, Optional, Union

from .iteration import windowed_queue_iter
from .position import find_element_in_list


def sent_seg(
    text: str,
    special_seg_indicators: Optional[List] = None,
    lang: Optional[str] = "zh",
    punctuations: Union[MutableSet[str], None] = None,
    quotation_seg_mode: Optional[bool] = True,
) -> List[str]:
    """Cut texts into sentences (in Chinese or English).

    Args:
        text: texts ready to be cut
        special_seg_indicators: some special segment indicators and
            their replacement ( [indicator, replacement] ), in baike data,
            this argument could be ``[('###', '\\n'), ('%%%', ' '), ('%%', ' ')]``
        lang: languages that your corpus is, support ``zh`` for Chinese
            and ``en`` for English now.
        punctuations: you can split the texts by specified punctuations.
            texts will not be splited by ``;``, so you can specify them by your own.
        quotation_seg_mode: if True, the quotations will be regarded as a
            part of the former sentence.
            e.g. ``我说：“翠花，上酸菜。”她说：“欸，好嘞。”``
            the text will be splited into
            ``['我说：“翠花，上酸菜。”', '她说：“欸，好嘞。”']``, other than
            ``['我说：“翠花，上酸菜。', '”她说：“欸，好嘞。”']``

    Returns:
        a list of strings, which are splited sentences.

    Raises:
        ValueError: if text is not string
    """
    # if texts are not in string format, raise an error
    if not isinstance(text, str):
        raise ValueError

    # if the text is empty, return a list with an empty string
    if len(text) == 0:
        return []

    text_return = text

    # segment on specified indicators
    # special indicators standard, like [('###', '\n'), ('%%%', '\t'), ('\s', '')]
    if special_seg_indicators:
        for indicator in special_seg_indicators:
            text_return = re.sub(indicator[0], indicator[1], text_return)

    if lang == "zh":
        punkt = {"。", "？", "！", "…"}
    elif lang == "en":
        punkt = {".", "?", "!"}
    if punctuations:
        punkt = punkt | punctuations

    if quotation_seg_mode:
        text_return = re.sub(
            "([%s]+[’”`'\"]*)" % ("".join(punkt)), "\\1\n", text_return
        )
    else:
        text_return = re.sub("([{}])".format("".join(punkt)), "\\1\n", text_return)

    # drop sentences with no length
    return [
        s.strip()
        for s in filter(
            lambda x: len(x.strip()) == 1
            and x.strip() not in punkt
            or len(x.strip()) > 0,
            text_return.split("\n"),
        )
    ]


def split_list_by_element(
    elements: List[Any], delimiter: Any, keep_empty_segments: bool = False
) -> List[List[Any]]:
    res = []
    pos = find_element_in_list(elements, delimiter)
    if not pos:
        return [elements]
    last_index = len(elements)
    if last_index not in pos:
        pos.append(last_index)
    res.append(elements[: pos[0]])
    for batch in windowed_queue_iter(pos, 2, 1, drop_last=False):
        if len(batch) == 2:
            res.append(elements[batch[0] + 1 : batch[1]])

    if keep_empty_segments:
        return res
    else:
        return list(filter(lambda el: el, res))


def split_data_with_portion(
    data: list, ratios: List[float], shuffle: bool = False
) -> List[list]:
    """
    Split data into multiple portions, if sum(ratios) < 1.0, then the data will be cut into `len(ratios) + 1` portions.
    """
    if shuffle:
        random.shuffle(data)
    tot_num = len(data)
    split = []
    for r in ratios:
        portion_len = int(tot_num * r)
        portion = data[:portion_len]
        split.append(portion)
        data = data[portion_len:]
    if len(data) > 0:
        split.append(data)
    return split
