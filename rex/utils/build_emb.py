from typing import Callable, List

from rex.data.vocab import Vocab
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.io import load_embedding_file, load_line_iterator
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar


def build_vocab(tokenize_func: Callable, lines: List[str]):
    vocab = set()
    for line in lines:
        tokens: List[str] = tokenize_func(line.strip())
        vocab.update(tokens)
    return vocab


def build_emb(
    raw_emb_filepath: str,
    dump_emb_filepath: str,
    *files: str,
    tokenize_func: Callable = None,
):
    """build vocab and filter out useless embeddings from ``raw_emb_filepath``"""
    logger.info(f"loading embedding from {raw_emb_filepath}")
    tokens, token2vec = load_embedding_file(raw_emb_filepath)
    logger.info(f"#tokens in emb file: {len(tokens)}")

    if tokenize_func is None:
        logger.info("Use char tokenize function.")
        tokenize_func = list

    logger.info("building vocab")
    raw_vocab = set()
    for filepath in pbar(files):
        for line in load_line_iterator(filepath):
            ins_tokens = tokenize_func(line.strip())
            raw_vocab.update(ins_tokens)
    logger.info(f"vocab size of corpora: {len(raw_vocab)}")

    logger.info("updating vocab")
    final_tokens = raw_vocab & set(tokens)
    logger.info(f"final vocab size of the intersection: {len(final_tokens)}")
    final_vocab = Vocab(init_pad_unk_emb=True)
    for token in pbar(final_tokens):
        final_vocab.add(token, token2vec[token])

    logger.info("dumping vocab")
    final_vocab.save_pretrained(dump_emb_filepath, dump_weights=True)


if __name__ == "__main__":
    config = ConfigParser.parse_cmd(
        ConfigArgument("-r", "--raw-emb-filepath", help="raw embedding filepath"),
        ConfigArgument("-o", "--dump-emb-filepath", help="dumped embedding filepath"),
        ConfigArgument("-f", "--filepaths", nargs="*", help="raw text filepaths"),
    )
    build_emb(config.raw_emb_filepath, config.dump_emb_filepath, *config.filepaths)
