import argparse

from dante_parser.data import load_data, load_splitted_data
from dante_parser.data.conllu import write_conllu


def main():
    parser = argparse.ArgumentParser("Creates UDPipe2 training files")
    parser.add_argument("--datasets", type=str, default="bosque dante_01")
    parser.add_argument(
        "--all_data",
        default=False,
        action="store_true",
        help="Concatenate all sets (train, val, test) to train",
    )
    parser.add_argument(
        "--no_val",
        default=True,
        action="store_true",
        help="If set, only creates train and tests files.",
    )
    args = parser.parse_args()

    datasets = args.datasets.split()

    if args.all_data:
        train = load_data(datasets)
    else:
        train, test = load_splitted_data(datasets, args.no_val)
        print(len(train), len(test))
        write_conllu("test.conllu", test)

    write_conllu("train.conllu", train)


if __name__ == "__main__":
    main()
