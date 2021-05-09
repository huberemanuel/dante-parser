import os

import dante_parser


def get_datasets():
    """ Return list of supported datasets with corresponding path and filetypes """
    
    base_path = os.path.dirname(dante_parser.__file__)
    base_path = os.path.join(base_path, "datasets")
    datasets = {
        "bosque": {
            "train": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-train.txt"),
                "filetype": "conllu"
            },
            "dev": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-dev.txt"),
                "filetype": "conllu"
            },
            "test": {
                "path": os.path.join(base_path, "bosque", "pt_bosque-ud-test.txt"),
                "filetype": "conllu"
            }
        },
        "dante_01": {
            "original": {
                "path": os.path.join(base_path, "dante_01", "tweets_stocks.csv"),
                "filetype": "csv"
            },
            "1a150": {
                "path": os.path.join(base_path, "dante_01", "1a150.csv"),
                "filetype": "conllu"
            }
        }
    }
    return datasets
