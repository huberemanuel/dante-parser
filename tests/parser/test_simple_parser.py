from typing import List
import pytest

from dante_parser.parser import SimpleParser


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
