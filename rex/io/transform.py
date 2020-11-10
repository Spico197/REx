from typing import Optional, List

import jieba
import thulac


class WordCutter(object):
    def __init__(self, package):
        """word segmentation

        Args:
            package: `jieba` or `thulac`
        """
        raise NotImplementedError


def find_all_positions(
    long_list: List[str],
    sub_list: List[str],
    add_end: Optional[bool] = True
) -> List:
    """find all sub list positions in list
    
    """