from collections import defaultdict
from typing import List, DefaultDict, Tuple


def get_entities_from_tag_seq(
    chars: List[str], tags: List[str]
) -> DefaultDict[str, List[Tuple]]:
    r"""
    get entities from a seqence of chars and tags, BIO and BMES schemas are both supported
    Args:
        chars: list of chars (single string)
        tags: list of tags (in string format), like `B`, `I-Subj`, `B_PER`.
            if there is no postfix type string, entities will not be categorized
    Returns:
        dict of list of tuples:
            {
                "default": [(entity name, entity type, start position, end position + 1), ...],
                "PER": [(entity name, entity type, start position, end position + 1)],
                ...
            }
    """
    if len(tags) > len(chars):
        tags = tags[: len(chars)]
    elif len(chars) > len(tags):
        chars = chars[: len(tags)]
    entities = defaultdict(list)
    last_type = ""
    ent = ""
    ent_start = -1

    for idx, (char, tag) in enumerate(zip(chars, tags)):
        if len(tag) > 2:
            curr_type = tag[2:]
        else:
            curr_type = "default"

        if tag.startswith("B"):
            if len(ent) > 0:
                entities[last_type].append((ent, last_type, ent_start, idx))

            ent = char
            last_type = curr_type
            ent_start = idx

        elif tag.startswith("I") or tag.startswith("M"):
            if curr_type == last_type:
                ent += char
            else:
                # illegal case, early stop here
                if len(ent) > 0:
                    entities[last_type].append((ent, last_type, ent_start, idx))
                ent = ""
                last_type = ""
                ent_start = -1

        elif tag.startswith("E"):
            if curr_type == last_type:
                ent += char

            if len(ent) > 0:
                entities[last_type].append((ent, last_type, ent_start, idx + 1))

            ent = ""
            last_type = ""
            ent_start = -1

        elif tag.startswith("S"):
            if len(ent) > 0:
                # last entity is not stopped
                entities[last_type].append((ent, last_type, ent_start, idx))

            entities[curr_type].append((char, curr_type, idx, idx + 1))

        else:
            # O
            if len(ent) > 0:
                entities[last_type].append((ent, last_type, ent_start, idx))

            ent = ""
            last_type = ""
            ent_start = -1

    if len(ent) > 0:
        entities[last_type].append((ent, last_type, ent_start, ent_start + len(ent)))

    return entities


def get_num_illegal_tags_from_tag_seq(tags: List[str]) -> int:
    r"""
    get number of illegal tags from a seqence of chars and tags, BIO and BMES schemas are both supported
    Args:
        tags: list of tags (in string format), like `B`, `I-Subj`, `B_PER`.
            if there is no postfix type string, entities will not be categorized
    Returns:
        number of illegal tags
    """
    num_illegal_tags = 0
    last_type = "default"
    last_tag = "O"

    for tag in tags:
        if len(tag) > 2:
            curr_type = tag[2:]
        else:
            curr_type = "default"

        if tag[0] in "IME":
            if last_tag not in "BIM" or curr_type != last_type:
                # illegal case
                num_illegal_tags += 1
        elif tag[0] == "S" and last_tag in "IM":
            num_illegal_tags += 1

        last_type = curr_type
        last_tag = tag[0]

    return num_illegal_tags
