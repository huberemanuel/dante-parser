import argparse
from typing import List

import pytest

from dante_parser.parser import ArcSystem, SimpleParser


def test_arc_system_basic():
    tokens = ["tudo", "bem", "?"]
    arc = ArcSystem(tokens)
    assert len(arc.stack) == 0
    assert len(arc.buffer) == (len(tokens) + 1)
    assert len(arc.deps) == 0

    for _ in range(len(tokens) + 1):
        arc.shift()
    assert len(arc.stack) == (len(tokens) + 1)
    assert arc.stack[0] == ArcSystem.ROOT
    assert len(arc.buffer) == 0
    assert len(arc.deps) == 0

    arc.left_arc()
    assert len(arc.deps) == 1
    assert len(arc.stack) == 3
    assert arc.stack[-1] == 3
    assert arc.deps[0] == (3, 2)
    arc.right_arc()
    assert len(arc.deps) == 2
    assert len(arc.stack) == 2
    assert arc.deps[1] == (1, 3)
    assert arc.stack[1] == 1
    assert arc.stack[-1] == 1

    assert not arc.left_arc()
    arc.right_arc()

    assert len(arc.stack) == 1
    assert len(arc.buffer) == 0
    assert len(arc.deps) == 3


@pytest.mark.parametrize(
    "text,n_sentences",
    [
        ("Isso é uma sentença. Isso é outra sentença.", 2),
        ("Olá, tudo bem? Sim e com você?", 2),
        ("Uma única sentença,", 1),
    ],
)
def test_sentence_segmentation(text: str, n_sentences: int):
    parser = SimpleParser()
    assert len(list(parser.sentence_segmentation(text))) == n_sentences


@pytest.mark.parametrize(
    "text,tokenized_text",
    [
        (
            "Isso é um token.",
            ["Isso", "é", "um", "token", "."],
        ),
        (
            "Olá, tudo bem?",
            ["Olá", ",", "tudo", "bem", "?"],
        ),
        ("Uma única sentença,", ["Uma", "única", "sentença", ","]),
    ],
)
def test_tokenization(text: str, tokenized_text: List[str]):
    parser = SimpleParser()
    assert list(parser.tokenize(text)) == tokenized_text


@pytest.mark.parametrize(
    "tokens,true_tags",
    (
        (["Alberto", "cantará", "amanhã"], ["PROPN", "VERB", "ADV"]),
        # (["Amanhã", "estarei", "livre"],
        # ["ADV", "VERB", "ADJ"]), pt_news_sm can't handle this poweful sentence.
    ),
)
def test_tagging(tokens: List[str], true_tags: List[str]):
    parser = SimpleParser()
    pred_tags = list(parser.tag(tokens))
    assert len(pred_tags) == len(tokens)
    assert pred_tags == true_tags


@pytest.mark.parametrize(
    "tokens,true_tags",
    (
        (["Alberto", "cantará", "amanhã"], ["Alberto", "cantar", "amanhã"]),
        (["Amanhã", "estarei", "livre"], ["Amanhã", "estar", "livrar"]),
    ),
)
def test_lemmatize(tokens: List[str], true_tags: List[str]):
    parser = SimpleParser()
    pred_lemmas = list(parser.lemmatize(tokens))
    assert len(pred_lemmas) == len(tokens)
    assert pred_lemmas == true_tags
