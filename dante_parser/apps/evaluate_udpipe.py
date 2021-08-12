import argparse

from dante_tokenizer.evaluate import evaluate_dataset
from ufal.udpipe import Model

from dante_parser.data.conllu import (
    extract_tags,
    read_conllu,
    remove_multiword_rows,
    remove_tags,
)
from dante_parser.parser.udpipe import predict_udpipe


def main():
    parser = argparse.ArgumentParser("Evaluate UDPipe model")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("input_conllu", type=str, help="Path to the evaluation CoNLL-U")
    args = parser.parse_args()

    model = Model.load(args.model_path)
    true = read_conllu(args.input_conllu)
    # Removing tags from input_conllu
    input_data = list(map(remove_tags, true))
    preds = predict_udpipe(input_data, model)

    # Removing multiword rows to evaluation
    preds = remove_multiword_rows(preds)
    true = remove_multiword_rows(true)

    pred_tags = extract_tags(preds)
    true_tags = extract_tags(true)

    assert len(pred_tags) == len(true_tags)
    for t, i in zip(true_tags, pred_tags):
        assert len(t) == len(i), "{}, {}".format(t, i)

    # Removing tags that aren't on the UD standard (e.g E_DIGIT, E_PROC)
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
    ]

    pred_tags = list(
        map(lambda x: list(map(lambda y: y if y in ud_tags else "X", x)), pred_tags)
    )
    true_tags = list(
        map(lambda x: list(map(lambda y: y if y in ud_tags else "X", x)), true_tags)
    )

    print(evaluate_dataset(pred_tags, true_tags))


if __name__ == "__main__":
    main()
