import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from rex.utils.logging import logger


def allowed_transitions(
    constraint_type: str, labels: Dict[int, str]
) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}.
    # Returns
    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [
        (start_tag, "START"),
        (end_tag, "END"),
    ]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(
                constraint_type, from_tag, from_entity, to_tag, to_entity
            ):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.
    # Returns
    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I")
                and to_tag in ("I", "L")
                and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def get_transition_mask_mat(
    scheme: str,
    id2label: Dict[int, str],
    return_start_end_transitions: Optional[bool] = True,
) -> Union[torch.Tensor, Tuple]:
    r"""get transition mask matrix for masked CRF

    Args:
        scheme: tagging label scheme, choices are "BIO", "IOB1", "BIOUL", and "BMES"
        id2label: tag id to tag name. e.g. {0: 'O', 1: 'B-PER', ...}
        return_start_end_transitions: whether to return start and end transition vectors

    Returns:
        transition matrix
        start transition vector if `return_start_end_transitions`
        end transition vector if `return_start_end_transitions`
    """
    allowed_with_start_end = allowed_transitions(scheme, id2label)
    return get_transition_mask_mat_from_allowed(
        len(id2label), allowed_with_start_end, return_start_end_transitions
    )


def get_transition_mask_mat_from_allowed(
    num_tags: int,
    allowed_with_start_end: List[Tuple],
    return_start_end_transitions: Optional[bool] = True,
):
    r"""get transition mask matrix for masked CRF given allowed_transitions

    Args:
        num_tags: number of tags (despite START and END)
        allowed_with_start_end: allowed transition tuples, got from `allowed_transitions()`
        return_start_end_transitions: whether to return start and end transition vectors

    Returns:
        transition matrix
        start transition vector if `return_start_end_transitions`
        end transition vector if `return_start_end_transitions`
    """
    trans_mask = -torch.ones(num_tags, num_tags, dtype=torch.float)
    start_mask = -torch.ones(num_tags, dtype=torch.float)
    end_mask = -torch.ones(num_tags, dtype=torch.float)

    start_tag_idx = num_tags
    end_tag_idx = num_tags + 1

    for from_label_index, to_label_index in allowed_with_start_end:
        if from_label_index == start_tag_idx and to_label_index == end_tag_idx:
            continue
        if from_label_index == start_tag_idx:
            start_mask[to_label_index] = 1.0
        elif to_label_index == end_tag_idx:
            end_mask[from_label_index] = 1.0
        else:
            trans_mask[from_label_index, to_label_index] = 1.0

    trans_mask *= 100
    start_mask *= 100
    end_mask *= 100

    if return_start_end_transitions:
        return trans_mask, start_mask, end_mask
    else:
        return trans_mask


"""
Plain Conditional Random Field
This CRF is implemented in:
    pytorch-crf: https://github.com/kmkurn/pytorch-crf
    pytorch-crf version = '0.7.2'
"""


class PlainCRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        logits: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            logits (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(logits, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            logits = logits.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(logits, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(logits, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(
        self, logits: torch.Tensor, mask: Optional[torch.LongTensor] = None
    ) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            logits (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(logits, mask=mask)
        if mask is None:
            mask = logits.new_ones(logits.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            logits = logits.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(logits, mask)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.LongTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.LongTensor
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


"""
Masked Conditional Random Field
This CRF is officially implemented in:
    MaskedCRF: https://github.com/DandyQi/MaskedCRF/blob/master/masked_crf.py
"""


class MaskedCRF(nn.Module):
    """Masked Conditional random field.
    Args:
        num_tags: Number of tags.
        constraints: Constraints got from `allowed_transitions()`
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
        masked_decoding: Whether using allowed transition mask while path decoding
        masked_training: Whether using allowed transition mask while training
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    """

    def __init__(
        self,
        num_tags: int,
        constraints: List[Tuple],
        batch_first: bool = True,
        masked_decoding: Optional[bool] = True,
        masked_training: Optional[bool] = True,
    ) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.masked_decoding = masked_decoding
        self.masked_training = masked_training

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        (
            self.trans_mask,
            self.start_trans_mask,
            self.end_trans_mask,
        ) = get_transition_mask_mat_from_allowed(
            num_tags, constraints, return_start_end_transitions=True
        )
        self.trans_mask = nn.Parameter(self.trans_mask, requires_grad=False)
        self.start_trans_mask = nn.Parameter(self.start_trans_mask, requires_grad=False)
        self.end_trans_mask = nn.Parameter(self.end_trans_mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        logits: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            logits (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(logits, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            logits = logits.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if self.masked_training:
            transitions, start_transitions, end_transitions = self.get_min_mask()
        else:
            transitions, start_transitions, end_transitions = (
                self.transitions,
                self.start_transitions,
                self.end_transitions,
            )

        # shape: (batch_size,)
        numerator = self._compute_score(
            transitions, start_transitions, end_transitions, logits, tags, mask
        )
        # shape: (batch_size,)
        denominator = self._compute_normalizer(
            transitions, start_transitions, end_transitions, logits, mask
        )
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(
        self, logits: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            logits (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(logits, mask=mask)
        if mask is None:
            mask = logits.new_ones(logits.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            logits = logits.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if self.masked_decoding:
            transitions, start_transitions, end_transitions = self.get_min_mask()
        else:
            transitions, start_transitions, end_transitions = (
                self.transitions,
                self.start_transitions,
                self.end_transitions,
            )

        return self._viterbi_decode(
            transitions, start_transitions, end_transitions, logits, mask
        )

    def get_min_mask(self):
        masked_transitions = torch.min(self.transitions, self.trans_mask)
        masked_start_transitions = torch.min(
            self.start_transitions, self.start_trans_mask
        )
        masked_end_transitions = torch.min(self.end_transitions, self.end_trans_mask)
        return masked_transitions, masked_start_transitions, masked_end_transitions

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self,
        transitions,
        start_transitions,
        end_transitions,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        transitions,
        start_transitions,
        end_transitions,
        emissions: torch.Tensor,
        mask: torch.ByteTensor,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self,
        transitions,
        start_transitions,
        end_transitions,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


"""
Constraint Conditional Random Field
This CRF is implemented in:
    AllenNLP: https://github.com/allenai/allennlp/blob/main/allennlp/modules/conditional_random_field.py
"""

VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score


class ConstraintCRF(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.
    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf
    # Parameters
    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `decode()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_tags: int,
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(
        self, logits: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = torch.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return torch.logsumexp(stops, -1)

    def _joint_likelihood(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(
            1, last_tags.view(-1, 1)
        )  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
        self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """

        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool)

        log_denominator = self._input_likelihood(logits, mask)
        log_numerator = self._joint_likelihood(logits, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    @classmethod
    def viterbi_decode(
        cls,
        tag_sequence: torch.Tensor,
        transition_matrix: torch.Tensor,
        tag_observations: Optional[List[int]] = None,
        allowed_start_transitions: torch.Tensor = None,
        allowed_end_transitions: torch.Tensor = None,
        top_k: int = None,
    ):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.
        # Parameters
        tag_sequence : `torch.Tensor`, required.
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.
        transition_matrix : `torch.Tensor`, required.
            A tensor of shape (num_tags, num_tags) representing the binary potentials
            for transitioning between a given pair of tags.
        tag_observations : `Optional[List[int]]`, optional, (default = `None`)
            A list of length `sequence_length` containing the class ids of observed
            elements in the sequence, with unobserved elements being set to -1. Note that
            it is possible to provide evidence which results in degenerate labelings if
            the sequences of tags you provide as evidence cannot transition between each
            other, or those transitions are extremely unlikely. In this situation we log a
            warning, but the responsibility for providing self-consistent evidence ultimately
            lies with the user.
        allowed_start_transitions : `torch.Tensor`, optional, (default = `None`)
            An optional tensor of shape (num_tags,) describing which tags the START token
            may transition *to*. If provided, additional transition constraints will be used for
            determining the start element of the sequence.
        allowed_end_transitions : `torch.Tensor`, optional, (default = `None`)
            An optional tensor of shape (num_tags,) describing which tags may transition *to* the
            end tag. If provided, additional transition constraints will be used for determining
            the end element of the sequence.
        top_k : `int`, optional, (default = `None`)
            Optional integer specifying how many of the top paths to return. For top_k>=1, returns
            a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
            tuple with just the top path and its score (not in lists, for backwards compatibility).
        # Returns
        viterbi_path : `List[int]`
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : `torch.Tensor`
            The score of the viterbi path.
        """
        if top_k is None:
            top_k = 1
            flatten_output = True
        elif top_k >= 1:
            flatten_output = False
        else:
            raise ValueError(
                f"top_k must be either None or an integer >=1. Instead received {top_k}"
            )

        sequence_length, num_tags = list(tag_sequence.size())

        has_start_end_restrictions = (
            allowed_end_transitions is not None or allowed_start_transitions is not None
        )

        if has_start_end_restrictions:

            if allowed_end_transitions is None:
                allowed_end_transitions = torch.zeros(num_tags)
            if allowed_start_transitions is None:
                allowed_start_transitions = torch.zeros(num_tags)

            num_tags = num_tags + 2
            new_transition_matrix = torch.zeros(num_tags, num_tags)
            new_transition_matrix[:-2, :-2] = transition_matrix

            # Start and end transitions are fully defined, but cannot transition between each other.

            allowed_start_transitions = torch.cat(
                [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
            )
            allowed_end_transitions = torch.cat(
                [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
            )

            # First define how we may transition FROM the start and end tags.
            new_transition_matrix[-2, :] = allowed_start_transitions
            # We cannot transition from the end tag to any tag.
            new_transition_matrix[-1, :] = -math.inf

            new_transition_matrix[:, -1] = allowed_end_transitions
            # We cannot transition to the start tag from any tag.
            new_transition_matrix[:, -2] = -math.inf

            transition_matrix = new_transition_matrix

        if tag_observations:
            if len(tag_observations) != sequence_length:
                raise ValueError(
                    "Observations were provided, but they were not the same length "
                    "as the sequence. Found sequence of length: {} and evidence: {}".format(
                        sequence_length, tag_observations
                    )
                )
        else:
            tag_observations = [-1 for _ in range(sequence_length)]

        if has_start_end_restrictions:
            tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
            zero_sentinel = torch.zeros(1, num_tags)
            extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
            tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
            tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
            sequence_length = tag_sequence.size(0)

        path_scores = []
        path_indices = []

        if tag_observations[0] != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[tag_observations[0]] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[0, :].unsqueeze(0))

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            summed_potentials = (
                path_scores[timestep - 1].unsqueeze(2) + transition_matrix
            )
            summed_potentials = summed_potentials.view(-1, num_tags)

            # Best pairwise potential path score from the previous timestep.
            max_k = min(summed_potentials.size()[0], top_k)
            scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

            # If we have an observation for this timestep, use it
            # instead of the distribution over tags.
            observation = tag_observations[timestep]
            # Warn the user if they have passed
            # invalid/extremely unlikely evidence.
            if tag_observations[timestep - 1] != -1 and observation != -1:
                if (
                    transition_matrix[tag_observations[timestep - 1], observation]
                    < -10000
                ):
                    logger.warning(
                        (
                            "The pairwise potential between tags you have passed as "
                            "observations is extremely unlikely. Double check your evidence "
                            "or transition potentials!"
                        )
                    )
            if observation != -1:
                one_hot = torch.zeros(num_tags)
                one_hot[observation] = 100000.0
                path_scores.append(one_hot.unsqueeze(0))
            else:
                path_scores.append(tag_sequence[timestep, :] + scores)
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        path_scores_v = path_scores[-1].view(-1)
        max_k = min(path_scores_v.size()[0], top_k)
        viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
        viterbi_paths = []
        for i in range(max_k):
            viterbi_path = [best_paths[i]]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
            # Reverse the backward path.
            viterbi_path.reverse()

            if has_start_end_restrictions:
                viterbi_path = viterbi_path[1:-1]

            # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)

        if flatten_output:
            return viterbi_paths[0], viterbi_scores[0]

        return viterbi_paths, viterbi_scores

    def decode(
        self,
        logits: torch.Tensor,
        mask: torch.BoolTensor = None,
        top_k: int = None,
        return_viterbi_score: Optional[bool] = False,
    ) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)
        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask[
            :num_tags, :num_tags
        ] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[
                start_tag, :num_tags
            ] = self.start_transitions.detach() * self._constraint_mask[
                start_tag, :num_tags
            ].data + -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[
                :num_tags, end_tag
            ] = self.end_transitions.detach() * self._constraint_mask[
                :num_tags, end_tag
            ].data + -10000.0 * (
                1 - self._constraint_mask[:num_tags, end_tag].detach()
            )
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - self._constraint_mask[:num_tags, end_tag].detach()
            )

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1 : (sequence_length + 1), :num_tags] = masked_prediction
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to `viterbi_decode`.
            viterbi_paths, viterbi_scores = self.viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                if not return_viterbi_score:
                    top_k_paths.append(viterbi_path)
                else:
                    top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths
