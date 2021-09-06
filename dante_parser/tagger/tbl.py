"""
Transition-based Learning Tagger adapted from: github.com/dhwaniraval/Brill_Tagger

Added support to CoNLL-U treebanks.
Added support to Universal Dependencies tag set.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List

from conllu import parse
from conllu.models import TokenList


# A class to hold the information about the tuples
class TaggerTuple:
    def __init__(self, from_tag, to_tag, pre_tag, score):
        self.from_tag = from_tag
        self.to_tag = to_tag
        self.pre_tag = pre_tag
        self.score = score


# A function to store the given corpus into string
def read_file(filename: str):
    with open(filename, "r") as in_f:
        return parse(in_f.read())


# Tokenize input file and create a unigram model
def tokenize(corpus_line: List[TokenList], output_path: str):
    unigram = defaultdict(dict)
    unigram_tokens = {}

    unigram_file = open(os.path.join(output_path, "unigram.txt"), "w")
    unigram_tokens_file = open(os.path.join(output_path, "unigram_tokens.txt"), "w")

    for sent in corpus_line:
        for word in sent:
            if not isinstance(word["id"], int):
                continue
            form = word["form"]
            tag = word["upos"]
            if form not in unigram.keys():
                unigram[form] = {}
            if tag not in unigram[form]:
                unigram[form][tag] = 0
            unigram[form][tag] += 1

            if form not in unigram_tokens.keys():
                unigram_tokens[form] = 0
            unigram_tokens[form] += 1

    for key, value in unigram.items():
        unigram_file.write(key + " " + str(value) + "\n")
    unigram_file.close()

    for key, value in unigram_tokens.items():
        unigram_tokens_file.write(key + " " + str(value) + "\n")
    unigram_tokens_file.close()

    return unigram


# Initialize the dummy corpus with mostly like tags
def initialize_with_most_likely_tag(
    unigram: Dict[str, Dict[str, int]], output_path: str
):
    most_likely_unigram = {}

    with open(
        os.path.join(output_path, "most_probable_unigram.txt"), "w"
    ) as most_likely_unigram_file:
        for key, value in unigram.items():
            sorted_list = sorted(value, key=value.get, reverse=True)
            most_likely_unigram[key] = sorted_list[0]
            most_likely_unigram_file.write(key + " " + str(sorted_list[0]) + "\n")
    most_likely_unigram_file.close()

    return most_likely_unigram


# Train the model to generate 10 transformational templates
def tbl(
    most_likely_unigram,
    corpus_tuple,
    correct_tag,
    current_tag,
    tag_set: List[str],
    output_path: str,
):
    n = 1
    transforms_queue = []

    while n <= 10:
        best_transform = get_best_transform(
            most_likely_unigram, corpus_tuple, correct_tag, current_tag, n, tag_set
        )

        if best_transform.from_tag == "" and best_transform.to_tag == "":
            break

        apply_transform(best_transform, corpus_tuple, current_tag, n, output_path)
        transforms_queue.append(best_transform)
        n = n + 1

    return transforms_queue


# A function to get the best transform
def get_best_transform(
    most_likely_unigram, corpus_tuple, correct_tag, current_tag, n, tag_set: List[str]
):
    instance = get_best_instance(
        most_likely_unigram, corpus_tuple, correct_tag, current_tag, n, tag_set
    )
    return instance


# A function to get the best instance
def get_best_instance(
    most_likely_unigram,
    corpus_tuple,
    correct_tag,
    current_tag,
    iteration,
    tag_set: List[str],
):
    best_score = 0
    all_tags = tag_set

    transform = TaggerTuple("", "", "", "")

    print("Iteration :: " + str(iteration))

    for from_tag in all_tags:
        for to_tag in all_tags:
            max_difference = 0
            num_good_transform = {}
            num_bad_transform = {}

            if from_tag == to_tag:
                continue

            for pos in range(1, len(corpus_tuple)):

                if to_tag == correct_tag[pos] and from_tag == current_tag[pos]:
                    rule = (current_tag[pos - 1], from_tag, to_tag)

                    if rule in num_good_transform:
                        num_good_transform[rule] += 1
                    else:
                        num_good_transform[rule] = 1
                elif from_tag == correct_tag[pos] and from_tag == current_tag[pos]:
                    rule = (current_tag[pos - 1], from_tag, to_tag)

                    if rule in num_bad_transform:
                        num_bad_transform[rule] += 1
                    else:
                        num_bad_transform[rule] = 1

            for key, value in num_good_transform.items():
                if key in num_bad_transform:
                    difference = num_good_transform[key] - num_bad_transform[key]
                else:
                    difference = num_good_transform[key]

                if difference > max_difference:
                    arg_max = key[0]
                    max_difference = difference

            if max_difference > best_score:
                best_rule = (
                    "Change tag FROM :: '"
                    + from_tag
                    + "' TO :: '"
                    + to_tag
                    + "' PREV tag :: '"
                    + arg_max
                    + "'"
                )
                best_score = max_difference

                print("Best Rule :: " + best_rule)
                transform = TaggerTuple(from_tag, to_tag, arg_max, best_score)

    return transform


# Apply transform after calculating best score of transformation template
def apply_transform(best_transform, corpus_tuple, current_tag, n, output_path: str):
    current_tag_File = open(
        os.path.join(output_path, "iteration_{}.txt".format(n)), "w"
    )

    for pos in range(1, len(corpus_tuple)):
        if (current_tag[pos] == best_transform.from_tag) and (
            current_tag[pos - 1] == best_transform.pre_tag
        ):
            current_tag[pos] = best_transform.to_tag

    for pos in range(0, len(current_tag)):
        current_tag_File.write(current_tag[pos] + "\n")


# Divide the corpus into 3 forms
# corpus_tuple : all the corpus words
# correct_tag :  all the corpus tags
# current_tag_File : most likely tag applied to all the words in corpus
def create_corpus_tuple(
    corpus_line: List[TokenList], most_likely_unigram, output_path: str
):
    corpus_tuple = []
    correct_tag = []
    current_tag = []

    corpus_tuple_file = open(os.path.join(output_path, "corpus_tuple.txt"), "w")
    correct_tag_file = open(os.path.join(output_path, "correct_tag.txt"), "w")
    current_tag_file = open(os.path.join(output_path, "current_tag.txt"), "w")

    for sent in corpus_line:
        for word in sent:
            if not isinstance(word["id"], int):
                continue
            form = word["form"]
            tag = word["upos"]

            corpus_tuple.append(form)
            correct_tag.append(tag)
            current_tag.append(most_likely_unigram[form])

            corpus_tuple_file.write(form + "\n")
            correct_tag_file.write(tag + "\n")
            current_tag_file.write(most_likely_unigram[form] + "\n")

    return corpus_tuple, correct_tag, current_tag


# sort all the transformation generated in oprder of their score
def sort_transformation_in_order_of_score(transformation_transforms_queue, output_path):
    sorted_Templates = sorted(
        transformation_transforms_queue, key=lambda x: x.score, reverse=True
    )
    index = 1

    with open(os.path.join(output_path, "top10.txt"), "w") as top10_file:
        for transformation in sorted_Templates:
            result = (
                str(index)
                + ":: From '"
                + transformation.from_tag
                + "' To '"
                + transformation.to_tag
                + "' when Prev '"
                + transformation.pre_tag
                + "'"
            )
            print(result)
            top10_file.write(result + "\n")
            index = index + 1
    top10_file.close()

    return sorted_Templates


def clean_tags(data: TokenList, tag_set: List[str]) -> TokenList:
    for sent in data:
        for word in sent:
            if isinstance(word["id"], int) and word["upos"] not in tag_set:
                word["upos"] = "X"
    return data


def main():
    parser = argparse.ArgumentParser("Train a TBL-tagger on given dataset and evaluate")
    parser.add_argument("--train_conllu", type=str, help="Path to the train CoNLL-U")
    parser.add_argument("--test_conllu", type=str, help="Path to the test CoNLL-U")
    args = parser.parse_args()
    output_path = "out"
    tag_set = [
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

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    train_sents = read_file(args.train_conllu)
    train_sents = clean_tags(train_sents, tag_set)
    unigram = tokenize(train_sents, output_path)

    most_likely_unigram = initialize_with_most_likely_tag(unigram, output_path)
    corpus_tuple, correct_tag, current_tag = create_corpus_tuple(
        train_sents, most_likely_unigram, output_path
    )
    transformation_transforms_queue = tbl(
        most_likely_unigram,
        corpus_tuple,
        correct_tag,
        current_tag,
        tag_set,
        output_path,
    )

    print("\n================== Top 10 Rules ==================")
    res = sort_transformation_in_order_of_score(
        transformation_transforms_queue, output_path
    )
    # TODO: Predict data on test set and evaluate metrics.


if __name__ == "__main__":
    main()
