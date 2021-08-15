from typing import Generator, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pkg_resources

import numpy as np
import json
from io import open
import os
import time
import argparse
import random
import conllu
from tqdm import tqdm

import spacy

from dante_parser.parser.simple_oracle.model import TransitionBasedDependencyParsing
from dante_parser.parser.simple_oracle.archybrid import Configuration


def train(model, optimizer, sentences):
    epoch_loss = 0.0
    epoch_arc_error = 0
    epoch_arc_cnt = 0
    sentence_cnt = 0

    random.shuffle(sentences)

    # Sets the is_training internal state to True
    model.train()

    for sentence in sentences:

        optimizer.zero_grad()

        # mlosses, mloss, err = model(sentence)
        mlosses, mloss = model(sentence)

        epoch_loss += mloss
        # epoch_arc_error += err
        epoch_arc_cnt += len(sentence)

        if len(mlosses):
            mlosses = sum(mlosses)
            mlosses.backward()
            optimizer.step()
        ##############
        # Here we can use footnote 8 in Eli's original paper
        ##############

        sentence_cnt += 1
        # if sentence_cnt % 5 == 0:
        #     print(
        #         "sentcnt:",
        #         sentence_cnt,
        #         "mloss:",
        #         epoch_loss,  #  , "err:", epoch_arc_error
        #     )

    return epoch_loss / epoch_arc_cnt, epoch_arc_error / epoch_arc_cnt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_data(path: str) -> Generator:
    file_data = open(path, "r", encoding="utf-8").read()
    return conllu.parse(file_data)


def add_emb(data: List[conllu.TokenList]):
    nlp = spacy.load("pt_core_news_lg")
    nlp.enable_pipe("tok2vec")

    for sent in tqdm(data):
        for tok in sent:
            tok.__dict__["emb"] = nlp(tok["form"])
    return data


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
    ]  # Should I include _ for multiword cases?
    POS_DIC = {k: v for v, k in enumerate(ud_tags)}
    # Hyper parameters
    BATCH_SIZE = 8
    # TODO: Use a pre-trained model and use this parameter
    EMB_DIM = 300  #  300
    POS_DIM = 32
    HIDDEN_DIM = 200
    HIDDEN_DIM2 = 100
    N_LAYERS = 2
    DROPOUT = 0.5

    training_data = load_data(args.train_conllu)[:300]
    training_data = add_emb(training_data)
    test_data = load_data(args.test_conllu)
    test_data = add_emb(test_data)

    model = TransitionBasedDependencyParsing(
        POS_DIC, EMB_DIM, POS_DIM, HIDDEN_DIM, HIDDEN_DIM2, N_LAYERS, DROPOUT
    )
    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda")
    model = model.to(device)

    N_EPOCHS = 10
    SAVE_DIR = "./models"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "hello_parsing.pt")
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_error = train(model, optimizer, training_data)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            "Epoch:",
            epoch + 1,
            "| Time:",
            epoch_mins,
            "m",
            epoch_secs,
            "s",
            "Train Loss:",
            train_loss,
        )

        # eval_loss, eval_error = evaluate(model, training_data)
        # print("Train Acc:", 1-eval_error)
        # eval_loss, eval_error = evaluate(model, dev_data)
        # print("Dev Acc:", 1 - eval_error)
        # if eval_error < best_dev_error:
        #     best_dev_error = eval_error
        #     torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
