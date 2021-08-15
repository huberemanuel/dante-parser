import argparse
import os
import pdb
import time
from operator import countOf
from pdb import set_trace
from typing import List

import conllu
import pkg_resources
import spacy
import torch
from torch import nn, optim, random


class SimpleOracle(nn.Module):
    def __init__(self, input_size=int):
        super().__init__()
        self.out = nn.Linear(input_size, 3)  # 3 transition states

    def forward(self, sentence):
        return self.out


class ArcSystem:
    """Basic arc system for projective arc system."""

    ROOT = 0
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3

    def __init__(self, tokens: List[str]) -> None:
        self.stack = []
        self.buffer = [ArcSystem.ROOT] + list(range(1, len(tokens) + 1))
        self.deps = []  # List of dependency relations

    def shift(self) -> bool:
        if len(self.buffer) < 1:
            return False  # Could not make a shift.
        else:
            element = self.buffer.pop(0)
            self.stack.append(element)
            return True

    def left_arc(self) -> bool:
        if len(self.stack) < 2:
            return False
        else:
            head = self.stack[-1]
            dep = self.stack.pop(-2)
            self.deps.append((head, dep))  # head -> dep
            return True

    def right_arc(self) -> bool:
        if len(self.stack) < 2:
            return False
        else:
            head = self.stack[-2]
            dep = self.stack.pop(-1)
            self.deps.append((head, dep))  # head -> dep
            return True


def load_data(path: str) -> List[conllu.TokenList]:
    file_data = open(path, "r", encoding="utf-8").read()
    return conllu.parse(file_data)


def main():
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    app_parser = argparse.ArgumentParser(
        "Train a Transition-based dependency parser model."
    )
    app_parser.add_argument(
        "--train_conllu",
        type=str,
        help="Path to the training CoNLL-U file.",
        default=pkg_resources.resource_filename(
            "dante_parser",
            os.path.join("datasets", "bosque", "pt_bosque-ud-train.conllu"),
        ),
    )
    app_parser.add_argument(
        "--test_conllu",
        type=str,
        help="Path to the input test CoNLL-U file.",
        default=pkg_resources.resource_filename(
            "dante_parser",
            os.path.join("datasets", "bosque", "pt_bosque-ud-test.conllu"),
        ),
    )
    args = app_parser.parse_args()

    ud_tags = [
        "ADJ",
        "ADV",
        "INTJ",
        "NOUN",
        "PROPN",
        "VERB",
        "ADP",
        "AUX",
        "CCONJ",
        "SCONJ",
        "DET",
        "NUM",
        "PART",
        "PRON",
        "PUNCT",
        "SYM",
        "X",
    ]
    POS_DIC = {k: v for v, k in enumerate(ud_tags)}
    N_EPOCHS = 2

    training_data = load_data(args.train_conllu)

    model = SimpleOracle(len(ud_tags))
    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    device = torch.device("cuda")
    model = model.to(device)

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # random.shuffle(training_data)

        model.train()
        train_loss = []
        for sentence in training_data:

            optimizer.zero_grad()

            # Transform input to array of ints
            # TODO: Filter multiword that have pos_ tags = "_"
            # import pdb

            # pdb.set_trace()
            input_data = list(map(lambda x: POS_DIC[x["upos"]], sentence))
            deps = list(map(lambda x: x["head"], sentence))

            sentence_loss = 0
            token_loss = 0

            arc = ArcSystem(input_data)
            arc.shift()
            transitions = []  # Oracle operations
            visiteds = []
            debug_i = 0
            while len(arc.buffer) > 0 or len(arc.stack) > 0:
                if len(arc.stack) < 2:
                    if len(arc.buffer) == 0:
                        break
                    if not arc.shift():
                        raise Exception("What?")
                        break
                    transitions.append(ArcSystem.SHIFT)
                elif sentence[arc.stack[-1] - 1]["head"] == arc.stack[-2] and countOf(
                    deps, arc.stack[-1]
                ) == countOf(
                    visiteds, arc.stack[-1]
                ):  # Check if all dependents from the top of the stack
                    # have been included before eliminating it.
                    if arc.right_arc():
                        transitions.append(ArcSystem.RIGHT_ARC)
                        visiteds.append(arc.deps[-1][0])
                elif (
                    sentence[arc.stack[-2] - 1]["head"] == arc.stack[-1]
                    and arc.stack[-2] != ArcSystem.ROOT
                ):
                    if arc.left_arc():
                        transitions.append(ArcSystem.LEFT_ARC)
                        visiteds.append(arc.deps[-1][0])
                else:
                    arc.shift()
                    transitions.append(ArcSystem.SHIFT)
                debug_i += 1

            for transition in transitions:
                # Quais serão as features do meu modelo? Ele tem que saber que estou
                # avançando no processo, precisa conhecer as palavras anteriores.
                pass

            logits = model(input_data)
            loss = nn.CrossEntropyLoss(logits, transitions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        end_time = time.time()
        # TODO: Average loss
        print(
            "Time elapsed for epoch {}: {} -> Total epoch loss {}".format(
                epoch, end_time - start_time, train_loss
            )
        )


if __name__ == "__main__":
    main()


class SimpleParser:
    def __init__(self) -> None:
        self.nlp = spacy.load("pt_core_news_sm")

    def sentence_segmentation(self, sentence: str) -> List[str]:
        """
        Uses Statistical sentence segmenter from Spacy.

        Parameters
        ----------
        sentences: List[str] or str
            Input text to be segmented.

        Returns
        -------
        List[str]
            List of segmenteted sentences
        """
        self.nlp.enable_pipe("senter")

        if not isinstance(sentence, str):
            raise ValueError("Input text is not instance of str ({})".format(sentence))

        return self.nlp(sentence).sents

    def tokenize(self, sentence: str) -> List[str]:
        """
        Uses default spacy tokenization.

        Parameters
        ----------
        sentence: str
            Input sentence.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(sentence, str):
            raise ValueError("Input text is not instance of str ({})".format(sentence))

        return map(lambda x: x.text, self.nlp(sentence))

    def tag(self, tokens: List[str]) -> List[str]:
        """
        Returning pos-tags.

        Parameters
        ----------
        tokens: List[str]
            Input sentence tokens.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(tokens, list):
            raise ValueError(
                "Input text is not instance of List[str] ({})".format(tokens)
            )

        tokens = " ".join(tokens)
        return map(lambda x: x.pos_, self.nlp(tokens))

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Returning lemmas.

        Parameters
        ----------
        tokens: List[str]
            Input sentence tokens.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(tokens, list):
            raise ValueError(
                "Input text is not instance of List[str] ({})".format(tokens)
            )

        tokens = " ".join(tokens)
        return map(lambda x: x.lemma_, self.nlp(tokens))
