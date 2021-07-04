import os
from random import shuffle

import dante_parser
from dante_parser.data.conllu import read_conllu
from dante_parser.data.sample import sents_train_test_split


def get_datasets():
    """Return list of supported datasets with corresponding path and filetypes"""

    base_path = os.path.dirname(dante_parser.__file__)
    base_path = os.path.join(base_path, "datasets")
    datasets = {
        "bosque": {
            "train": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-train.conllu"),
                "filetype": "conllu",
            },
            "dev": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-dev.conllu"),
                "filetype": "conllu",
            },
            "test": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-test.conllu"),
                "filetype": "conllu",
            },
        },
        "dante_01": {
            "original": {
                "path": os.path.join(base_path, "dante_01", "tweets_stocks.csv"),
                "filetype": "csv",
            },
            "1a147.conllu": {
                "path": os.path.join(base_path, "dante_01", "1a147.conllu"),
                "filetype": "conllu",
            },
            "dante_pack1.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack1.conllu"),
                "filetype": "conllu",
            },
            "dante_pack2.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack2.conllu"),
                "filetype": "conllu",
            },
            "dante_pack3.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack3.conllu"),
                "filetype": "conllu",
            },
            "dante_pack4.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack4.conllu"),
                "filetype": "conllu",
            },
            "dante_pack5.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack5.conllu"),
                "filetype": "conllu",
            },
            "dante_pack6.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack6.conllu"),
                "filetype": "conllu",
            },
            "dante_pack7.conllu": {
                "path": os.path.join(base_path, "dante_01", "dante_pack7.conllu"),
                "filetype": "conllu",
            },
        },
    }
    return datasets


def load_data(names: list, random_state: int = 42) -> list:
    """
    Load datasets from names list, joining all sets avaiable into a single list.

    Parameters
    ----------
    names: list
        List of dataset names.
    random_state: int
        Random seed.

    Returns
    -------
    list:
        List of sentences.
    """
    data = []
    datasets = get_datasets()

    for name in names:
        if name not in datasets.keys():
            raise ValueError(f"Dataset {name} not supported")

        for set_name, set_value in datasets[name].items():
            if set_value["filetype"] == "conllu":
                data += read_conllu(set_value["path"])

    return data


def load_splitted_data(
    names: list, only_test: bool = False, random_state: int = 42
) -> (list, list, list):
    """
    Load datasets from names list, returning train, val and test sets.
    If the dataset only have a train set, then it will sĺit it into train, val, test.

    Parameters
    ----------
    names: list
        List of datasets to be loaded.
    only_test: bool
        Doesn't return val set if True, else return all sets
    random_state: int
        Random seed.

    Returns
    -------
    (list, list, list):
        Train, Val and Test sets.
        Obs: If `only_test` is set, then it will return (list, list)
    """
    datasets = get_datasets()
    train_sents = []
    val_sents = []
    test_sents = []

    if not type(list) is list and type(list) is str:
        names = [names]

    for name in names:
        if name not in datasets.keys():
            raise ValueError(f"Dataset {name} not supported")

        for set_name, set_value in datasets[name].items():
            if set_name in ["val", "dev"]:
                if set_value["filetype"] == "conllu":
                    if only_test:
                        train_sents += read_conllu(set_value["path"])
                    else:
                        val_sents += read_conllu(set_value["path"])
            elif set_name == "test":
                if set_value["filetype"] == "conllu":
                    test_sents += read_conllu(set_value["path"])
            else:
                if set_name == "train" and set_value["filetype"] == "conllu":
                    train_sents += read_conllu(set_value["path"])
                elif set_value["filetype"] == "conllu":
                    train = read_conllu(set_value["path"])
                    train, test = sents_train_test_split(
                        train, random_state=random_state
                    )
                    if not only_test:
                        train, val = sents_train_test_split(
                            train, 0.1, random_sate=random_state
                        )
                        val_sents += val
                    train_sents += train
                    test_sents += test

    shuffle(train_sents)
    shuffle(val_sents)

    if only_test:
        return train_sents, test_sents
    else:
        return train_sents, val_sents, test_sents
