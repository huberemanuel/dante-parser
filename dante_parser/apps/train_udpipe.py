import argparse

from dante_parser.data import get_datasets, load_splitted_data, load_data
from dante_parser.parser.udpipe import train_udpipe


def main():
    parser = argparse.ArgumentParser("Trains Udpipe Model")
    parser.add_argument("--datasets", type=str, default="bosque dante_01")
    parser.add_argument("--all_data", default=False, action="store_true",
                        help="Concatenate all sets (train, val, test) to train")
    args = parser.parse_args()

    model_name = "mymodel.model"
    datasets = args.datasets.split(" ")
    if args.all_data:
        train = load_data(datasets)
        train_udpipe(train, [], model_name)
    else:
        train, val, test = load_splitted_data(datasets)

        train_udpipe(train, val, model_name)

if __name__ == "__main__":
    main()
