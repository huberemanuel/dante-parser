import argparse

from dante_parser.data import load_data, load_splitted_data
from dante_parser.data.conllu import read_conllu
from dante_parser.parser.udpipe import train_udpipe


def main():
    parser = argparse.ArgumentParser("Trains Udpipe Model")
    parser.add_argument("--datasets", type=str, default="bosque dante_01")
    parser.add_argument(
        "--all_data",
        default=False,
        action="store_true",
        help="Concatenate all sets (train, val, test) to train",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="mymodel.model",
        help="Name of the output model file",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to the input train CoNNL-U. If provied, it will ignore "
        + "`datases` and `all_data`",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(" ")
    if args.train_file:
        train = read_conllu(args.train_file)
        train_udpipe(train, [], args.out_name)
    elif args.all_data:
        train = load_data(datasets)
        train_udpipe(train, [], args.out_name)
    else:
        train, val, test = load_splitted_data(datasets)

        train_udpipe(train, val, args.out_name)


if __name__ == "__main__":
    main()
