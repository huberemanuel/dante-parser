import argparse
import logging
import math
import pickle
from typing import Dict, List

import numpy as np
import torch
from conllu import TokenList, parse
from torch import nn, optim
from tqdm import tqdm

from dante_parser.parser.first_parser.arc_system import ArcSystem
from dante_parser.parser.first_parser.model import FirstParser
from dante_parser.parser.first_parser.tree import DependencyTree
from dante_parser.parser.first_parser.utils import read_tree
from dante_parser.parser.first_parser.vocabulary import Vocabulary


def get_configuration_features(
    configuration: ArcSystem, vocabulary: Vocabulary
) -> List[int]:
    """
    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    Parameters
    ----------
    configuration: ArcSystem
        ArcSystem that stores the sentence.
    vocabulary: Vocabulary
        Given vocabulary.

    Returns
    -------
    List[int]
        List of features.
    """
    words = []
    posTags = []
    labels = []

    # Get the words and pos tags of the top 3 elements of the stack.
    for idx in range(2, -1, -1):
        stack = configuration.get_stack(idx)
        words.append(vocabulary.get_word_id(configuration.get_word(stack)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(stack)))

    # Get the words and pos tags of the top 3 elements of the buffer.
    for idx in range(3):
        buffer = configuration.get_buffer(idx)
        words.append(vocabulary.get_word_id(configuration.get_word(buffer)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(buffer)))

    # Get the words, labels, and pos tags of the first and second left
    # child and right child of the top two elements on the stack, and
    # Get the words, labels, and pos tags of the leftmost of the leftmost
    # and rightmost of the rightmost child of the top two elements on the stack
    for idx in range(2):
        stack = configuration.get_stack(idx)
        firstLeftChild = configuration.get_left_child(stack, 1)
        words.append(vocabulary.get_word_id(configuration.get_word(firstLeftChild)))
        labels.append(vocabulary.get_label_id(configuration.get_label(firstLeftChild)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(firstLeftChild)))

        firstRightChild = configuration.get_right_child(stack, 1)
        words.append(vocabulary.get_word_id(configuration.get_word(firstRightChild)))
        labels.append(vocabulary.get_label_id(configuration.get_label(firstRightChild)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(firstRightChild)))

        secondLeftChild = configuration.get_left_child(stack, 2)
        words.append(vocabulary.get_word_id(configuration.get_word(secondLeftChild)))
        labels.append(vocabulary.get_label_id(configuration.get_label(secondLeftChild)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(secondLeftChild)))

        secondRightChild = configuration.get_right_child(stack, 2)
        words.append(vocabulary.get_word_id(configuration.get_word(secondRightChild)))
        labels.append(
            vocabulary.get_label_id(configuration.get_label(secondRightChild))
        )
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(secondRightChild)))

        leftLeftChild = configuration.get_left_child(
            configuration.get_left_child(stack, 1), 1
        )
        words.append(vocabulary.get_word_id(configuration.get_word(leftLeftChild)))
        labels.append(vocabulary.get_label_id(configuration.get_label(leftLeftChild)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(leftLeftChild)))

        rightRightChild = configuration.get_right_child(
            configuration.get_right_child(stack, 1), 1
        )
        words.append(vocabulary.get_word_id(configuration.get_word(rightRightChild)))
        labels.append(vocabulary.get_label_id(configuration.get_label(rightRightChild)))
        posTags.append(vocabulary.get_pos_id(configuration.get_pos(rightRightChild)))

    features = []
    features += words + posTags + labels

    assert len(features) == 48
    return features


def generate_training_instances(
    transitions: List[str],
    sentences: List[TokenList],
    trees: List[DependencyTree],
    vocabulary: Vocabulary,
):
    """
    Generate traning instances, where X is the vectors of features
    of the given ArcSystem configuration and Y one-hot encoder
    representing the system action.
    Obs: Each sample is not a sentence, but a configuration of the
    ArcSystem, therefore, a whole sentence is a set of configurations.
    """
    instances = []
    for sentence, tree in tqdm(zip(sentences, trees)):
        if tree.is_projective():
            configuration = ArcSystem(sentence)
            configuration.shift()
            while not configuration.is_empty():

                oracle_decision = ArcSystem.get_oracle_decision(configuration, tree)
                feature = get_configuration_features(configuration, vocabulary)

                configuration.apply(oracle_decision)

                label = []
                for j in range(len(transitions)):
                    t = transitions[j]
                    if t == oracle_decision:
                        label.append(1.0)
                    elif configuration.can_apply(t):
                        label.append(0.0)
                    else:
                        label.append(-1.0)
                if 1.0 not in label:
                    logging.error(
                        "No gold_transition was for for sentence {} in "
                        "configuration: {}".format(sentence, configuration)
                    )

                instances.append({"input": feature, "label": label})
    return instances


def main():
    parser = argparse.ArgumentParser(
        "Training of a simple transition-based parsing with static oracle."
    )
    parser.add_argument(
        "--train_conllu", type=str, help="path to traning file on CoNLL-U format"
    )
    parser.add_argument(
        "--val_conllu", type=str, help="path to val file on CoNLL-U format"
    )
    parser.add_argument(
        "--log_level",
        default="error",
        type=str,
        help="Logging levels: info|warning|error",
    )
    parser.add_argument(
        "--cached_train",
        type=str,
        default=None,
        help="Path to pickled training instances.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    logger = logging.getLogger("Trainer")
    if args.log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "warning":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    train_sents = parse(open(args.train_conllu, "r").read())
    val_sents = parse(open(args.val_conllu, "r").read())

    train_trees = list(map(read_tree, train_sents))
    vocabulary = Vocabulary(train_sents, train_trees)

    dep_labels = [
        item[0]
        for item in sorted(vocabulary.label_token_to_id.items(), key=lambda k: k[1])
    ]
    transitions = ArcSystem.generate_transitions(dep_labels)

    assert len(transitions) == (len(dep_labels) * 2 + 1)
    logging.info(
        "Constructed {} possible dependency transitions".format(len(transitions))
    )

    if args.cached_train:
        train_instances = pickle.load(open(args.cached_train, "rb"))
    else:
        logging.info("Generating traning instances")
        train_instances = generate_training_instances(
            transitions, train_sents, train_trees, vocabulary
        )
        with open("train_instances.pickle", "wb") as train_file:
            pickle.dump(train_instances, train_file)

    kwargs = {
        "embedding_dim": 50,
        "n_features": len(train_instances[0]["input"]),
        "num_tokens": vocabulary.size(),
        "hidden_dim": 200,
        "num_transitions": len(transitions),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FirstParser(**kwargs).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.003,
    )
    loss_func = nn.CrossEntropyLoss()

    X_train = torch.tensor([x["input"] for x in train_instances], device=device)

    # A little gambiarra fow now:
    for i in range(len(train_instances)):
        train_instances[i]["label"] = np.argmax(train_instances[i]["label"])
        # for j in range(len(train_instances[i]["label"])):
        # train_instances[i]["label"][j] = max(0, train_instances[i]["label"][j])

    y_train = torch.tensor([x["label"] for x in train_instances], device=device)

    for epoch in range(args.epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, args.epochs))
        n_minibatches = math.ceil(len(train_instances) / args.batch_size)
        epoch_loss = 0

        permutation = torch.randperm(X_train.size()[0])
        with tqdm(total=(n_minibatches)) as prog:
            for i in range(0, len(train_instances), args.batch_size):
                optimizer.zero_grad()

                indices = permutation[i : i + args.batch_size]

                batch_x, batch_y = X_train[indices], y_train[indices]

                # in case you wanted a semi-full example
                outputs = model.forward(batch_x)
                loss = loss_func(outputs, batch_y)

                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
            prog.update(1)
        print("Epoch average loss: {}".format(np.mean(epoch_loss)))


if __name__ == "__main__":
    main()
