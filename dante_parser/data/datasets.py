import os

import dante_parser

def get_datasets():
    """ Return list of supported datasets """
    
    base_path = os.path.dirname(dante_parser.__file__)
    base_path = os.path.join(base_path, "datasets")
    datasets = {
        "bosque": {
            "train": os.path.join(base_path, "bosque", "pt_bosque-ud-train.txt"),
            "dev": os.path.join(base_path, "bosque", "pt_bosque-ud-dev.txt"),
            "test": os.path.join(base_path, "bosque", "pt_bosque-ud-test.txt"),
        }
    }
    return datasets
