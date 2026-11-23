"""Build CAP batches for word, constituency phrase, or random pooling."""

from collections import defaultdict
from functools import lru_cache
import random

import torch
from torch.nn.utils.rnn import pad_sequence


GRANULARITY_ALIASES = {
    "tw": "token_to_words",
    "tp": "token_to_phrases",
    "token_to_words": "token_to_words",
    "token_to_phrases": "token_to_phrases",
    "random": "random",
}


def normalise_granularity(value):
    try:
        return GRANULARITY_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"Unknown CAP granularity: {value!r}") from exc


def word_ranges(annotations, valid_length):
    ranges = [(int(span.start_idx), int(span.end_idx)) for span in annotations]
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start or end > valid_length:
            raise ValueError(
                f"CAP word spans must cover valid tokens without gaps or overlap; "
                f"expected start {cursor}, got {(start, end)}"
            )
        cursor = end
    if cursor != valid_length:
        raise ValueError(f"CAP word spans end at {cursor}, but attention mask has {valid_length} valid tokens")
    return ranges


def random_ranges(valid_length, seed, min_chunk=1, max_chunk=4):
    """Contiguous random chunks, matching original CAP's 1-4 token control."""
    rng = random.Random(seed)
    ranges = []
    start = 0
    while start < valid_length:
        end = min(start + rng.randint(min_chunk, max_chunk), valid_length)
        ranges.append((start, end))
        start = end
    return ranges


@lru_cache(maxsize=1)
def _phrase_parser():
    try:
        import benepar  # noqa: F401 - registers the spaCy component
        import spacy
    except ImportError as exc:
        raise RuntimeError("Phrase CAP needs benepar: pip install benepar") from exc
    nlp = spacy.load("en_core_web_sm")
    try:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except Exception as exc:
        raise RuntimeError(
            "Phrase CAP needs the benepar_en3 model; run "
            "python -c \"import benepar; benepar.download('benepar_en3')\""
        ) from exc
    return nlp


def merge_word_ranges(word_token_ranges, constituent_ids):
    """Merge adjacent words belonging to the same constituency node."""
    if len(word_token_ranges) != len(constituent_ids):
        raise ValueError("Each word span needs one constituent identifier")
    merged = []
    for (start, end), identifier in zip(word_token_ranges, constituent_ids):
        if merged and identifier is not None and identifier == merged[-1][2] and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end, identifier)
        else:
            merged.append((start, end, identifier))
    return [(start, end) for start, end, _ in merged]


def phrase_ranges(annotations, valid_length):
    """Group word spans by depth-2 Benepar constituency nodes."""
    from nltk.tree import Tree

    token_ranges = word_ranges(annotations, valid_length)
    words = [span.original_word for span in annotations]
    text = " ".join(words)
    word_chars = []
    offset = 0
    for word in words:
        word_chars.append((offset, offset + len(word)))
        offset += len(word) + 1

    doc = _phrase_parser()(text)
    token_constituents = {}
    for sentence_index, sentence in enumerate(doc.sents):
        tree = Tree.fromstring(sentence._.parse_string)
        if len(tree.leaves()) != len(sentence):
            raise ValueError("Benepar and spaCy token counts differ; cannot align CAP phrases")
        # Benepar may emit S directly or wrap it in TOP/ROOT. Select the
        # immediate sentence constituents (NP, VP, etc.) in either case.
        constituent_depth = 2 if tree.label() in {"TOP", "ROOT"} else 1
        for local_index, token in enumerate(sentence):
            path = tree.leaf_treeposition(local_index)
            token_constituents[token.i] = (sentence_index, path[:constituent_depth])

    word_constituents = []
    for word_index, (start, end) in enumerate(word_chars):
        ids = [token_constituents[token.i] for token in doc
               if start <= token.idx < end and token.i in token_constituents]
        word_constituents.append(ids[0] if ids else ("unparsed", word_index))
    return merge_word_ranges(token_ranges, word_constituents)


def create_cap_batches(dataloader, batch_size, granularity, seed, pad_token_id):
    """Regroup examples by the selected number of pooled positions."""
    granularity = normalise_granularity(granularity)
    groups = defaultdict(list)
    example_index = 0
    for source_batch in dataloader:
        for i, annotations in enumerate(source_batch["annotations"]):
            attention_mask = source_batch["attention_mask"][i]
            valid_length = int(attention_mask.sum().item())
            if valid_length == 0:
                raise ValueError("CAP cannot pool an empty input")
            if granularity == "token_to_words":
                ranges = word_ranges(annotations, valid_length)
            elif granularity == "token_to_phrases":
                ranges = phrase_ranges(annotations, valid_length)
            else:
                ranges = random_ranges(valid_length, seed + example_index)
            groups[len(ranges)].append({
                "input_ids": source_batch["input_ids"][i][:valid_length],
                "attention_mask": attention_mask[:valid_length],
                "labels": source_batch["labels"][i],
                "annotations": annotations,
                "ranges": ranges,
                "reduction_percentage": 100 * (valid_length - len(ranges)) / valid_length,
            })
            example_index += 1

    grouped_batches = {}
    batches_ranges = {}
    for length, items in groups.items():
        grouped_batches[length] = []
        batches_ranges[length] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            grouped_batches[length].append({
                "input_ids": pad_sequence([item["input_ids"] for item in chunk], batch_first=True,
                                          padding_value=pad_token_id),
                "attention_mask": pad_sequence([item["attention_mask"] for item in chunk], batch_first=True,
                                               padding_value=0),
                "labels": torch.stack([item["labels"] for item in chunk]),
                "annotations": [item["annotations"] for item in chunk],
                "reduction_percentage": [item["reduction_percentage"] for item in chunk],
            })
            batches_ranges[length].append([item["ranges"] for item in chunk])
    return grouped_batches, batches_ranges
