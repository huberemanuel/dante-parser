import argparse

from dante_parser.data import get_datasets, load_splitted_data
from dante_parser.parser.udpipe import train_udpipe


def main():
    parser = argparse.ArgumentParser("Trains Udpipe Model")
    parser.add_argument("--datasets", type=str, default="bosque dante_01")
    args = parser.parse_args()

    datasets = args.datasets.split(" ")
    train, val, test = load_splitted_data(datasets)

    train_udpipe(train, val, "mymodel.model")

if __name__ == "__main__":
    main()
