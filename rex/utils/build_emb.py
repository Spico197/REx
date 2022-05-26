from rex.data.vocab import Vocab
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.io import load_embedding_file, load_line_iterator
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar


def build_emb(raw_emb_filepath, dump_emb_filepath, *files):
    """build vocab and filter out useless embeddings from ``raw_emb_filepath``"""
    logger.info(f"loading embedding from {raw_emb_filepath}")
    tokens, token2vec = load_embedding_file(raw_emb_filepath)

    logger.info("building vocab")
    raw_vocab = set()
    for filepath in pbar(files):
        for line in load_line_iterator(filepath):
            raw_vocab.update(line.strip())

    logger.info("updating vocab")
    final_tokens = raw_vocab & set(tokens)
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
