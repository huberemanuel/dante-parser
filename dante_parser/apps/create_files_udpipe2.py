import argparse
import logging

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

    if not args.no_val:
        raise NotImplementedError("Validation split is currently not implement")

    if args.all_data:
        train = load_data(datasets)
        write_conllu("train.conllu", train)
    else:
        all_train = []
        print(datasets)
        for dataset in datasets:
            train, test = load_splitted_data([dataset], args.no_val)
            logging.info(
                "Loaded {} train and {} test sents for dataset {}".format(
                    len(train), len(test), dataset
                )
            )
            all_train += train
            write_conllu("{}_test.conllu".format(dataset), test)
        write_conllu("train.conllu", all_train)


if __name__ == "__main__":
    main()
