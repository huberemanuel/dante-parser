from typing import List

import pytest

from dante_parser.tagger import MostFrequentTagger


@pytest.mark.parametrize(
    "tags,expected_tags",
    (
        ([["NUM", "SYM"]], [["NUM", "SYM"]]),
        ([["PROPN", "NUM"], ["PROPN", "NUM"]], [["PROPN", "NUM"], ["PROPN", "NUM"]]),
        ([["ADV", "E_DIGIT"]], [["ADV", "X"]]),
        ([["PRON", "E_PPROC"]], [["PRON", "X"]]),
    ),
)
def test_clean_tags(tags: List[List[str]], expected_tags: List[List[str]]):
    clean_tags = MostFrequentTagger._clean_tags(tags)
    assert clean_tags == expected_tags


@pytest.mark.parametrize(
    "tokens,tags,query_words,expected_tags",
    (
        (
            [["#usim5", "Comprando", "8,61", "amanhã"]],
            [["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "amanhã"],
            ["PROPN", "VERB", "NUM", "ADV"],
        ),
        (
            [
                ["#usim5", "Comprando", "8,61", "amanhã"],
                ["#usim5", "Comprando", "8,61", "amanhã"],
            ],
            [["PROPN", "VERB", "NUM", "ADV"], ["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "amanhã"],
            ["PROPN", "VERB", "NUM", "ADV"],
        ),
        (
            [
                ["#usim5", "Comprando", "8,61", "Comprando"],
                ["#usim5", "Comprando", "8,61", "Comprando"],
            ],
            [["PROPN", "VERB", "NUM", "VERB"], ["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "Comprando"],
            ["PROPN", "VERB", "NUM", "VERB"],
        ),
    ),
)
def test_compute_counter(
    tokens: List[List[str]],
    tags: List[List[str]],
    query_words: List[str],
    expected_tags: List[str],
):
    counter = MostFrequentTagger._compute_counter(tokens, tags)

    for word, tag in zip(query_words, expected_tags):
        assert word in counter.keys()
        assert counter[word] == tag


@pytest.mark.parametrize(
    "tokens,tags,query_words,expected_tags",
    (
        (
            [["#usim5", "Comprando", "8,61", "amanhã"]],
            [["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "amanhã"],
            ["PROPN", "VERB", "NUM", "ADV"],
        ),
        (
            [
                ["#usim5", "Comprando", "8,61", "amanhã"],
                ["#usim5", "Comprando", "8,61", "amanhã"],
            ],
            [["PROPN", "VERB", "NUM", "ADV"], ["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "amanhã"],
            ["PROPN", "VERB", "NUM", "ADV"],
        ),
        (
            [
                ["#usim5", "Comprando", "8,61", "Comprando"],
                ["#usim5", "Comprando", "8,61", "Comprando"],
            ],
            [["PROPN", "VERB", "NUM", "VERB"], ["PROPN", "VERB", "NUM", "ADV"]],
            ["#usim5", "Comprando", "8,61", "Comprando"],
            ["PROPN", "VERB", "NUM", "VERB"],
        ),
    ),
)
def test_tagger(
    tokens: List[List[str]],
    tags: List[List[str]],
    query_words: List[str],
    expected_tags: List[str],
):
    tagger = MostFrequentTagger(tokens, tags)

    assert expected_tags, tagger.tag(query_words)
