from typing import List


class Vocabulary(object):
    """Vocabulary for char or words"""
    def __init__(self, texts: List[str], **kwargs):
        self.word2id = {}
        self.id2word = {}

    @classmethod
    def from_text_file(cls, filepath: str, **kwargs):
        raise NotImplementedError

    def get_word_id(self, word):
        raise NotImplementedError
