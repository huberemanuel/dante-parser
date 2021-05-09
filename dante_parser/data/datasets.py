import os
from random import shuffle

import dante_parser
from dante_parser.data.conllu import read_conllu
from dante_parser.data import sents_train_test_split


def get_datasets():
    """ Return list of supported datasets with corresponding path and filetypes """
    
    base_path = os.path.dirname(dante_parser.__file__)
    base_path = os.path.join(base_path, "datasets")
    datasets = {
        "bosque": {
            "train": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-train.conllu"),
                "filetype": "conllu"
            },
            "dev": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-dev.conllu"),
                "filetype": "conllu"
            },
            "test": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-test.conllu"),
                "filetype": "conllu"
            }
        },
        "dante_01": {
            "original": {
                "path": os.path.join(base_path, "dante_01", "tweets_stocks.csv"),
                "filetype": "csv"
            },
            "1a150": {
                "path": os.path.join(base_path, "dante_01", "1a150.conllu"),
                "filetype": "conllu"
            }
        }
    }
    return datasets

def load_data(names: list, random_state:int = 42) -> list:
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
        if not name in datasets.keys():
            raise ValueError(f"Dataset {name} not supported")
        
        for set_name, set_value in datasets[name].items():
            if set_value["filetype"] == "conllu":
                data += read_conllu(set_value["path"])

    return data

def load_splitted_data(names: list, random_state:int = 42) -> (list, list, list):
    """
    Load datasets from names list, returning train, val and test sets.

    Parameters
    ----------
    names: list
        List of datasets to be loaded.
    random_state: int
        Random seed.

    Returns
    -------
    (list, list, list):
        Train, Val and Test sets.
    """
    datasets = get_datasets()
    train_sents = []
    val_sents   = []
    test_sents  = []
    for name in names:
        if not name in datasets.keys():
            raise ValueError(f"Dataset {name} not supported")

        for set_name, set_value in datasets[name].items():
            if set_name in ["val", "dev"]:
                if set_value["filetype"] == "conllu":
                    val_sents += read_conllu(set_value["path"])
            elif set_name == "test":
                if set_value["filetype"] == "conllu":
                    test_sents += read_conllu(set_value["path"])
            else:
                if set_name == "train" and set_value["filetype"] == "conllu":
                    train_sents += read_conllu(set_value["path"])
                elif set_value["filetype"] == "conllu":
                    train = read_conllu(set_value["path"])
                    train, val = sents_train_test_split(train, random_state=random_state)
                    train_sents += train
                    val_sents += val
    shuffle(train_sents)
    shuffle(val_sents)
            
    return train_sents, val_sents, test_sents
