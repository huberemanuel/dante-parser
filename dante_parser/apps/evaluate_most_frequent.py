import argparse

from dante_tokenizer.evaluate import evaluate_dataset

from dante_parser.data import extract_tags, extract_tokens, read_conllu
from dante_parser.tagger import MostFrequentTagger


def pretty_metrics(metrics: tuple):
    for name, metric in zip(["Precision", "Recall", "F-Score"], metrics):
        print("{}: {:.2f} ± {:.2f}".format(name, *metric))


def main():
    parser = argparse.ArgumentParser("Evaluate Most Frequent Tagger")
    parser.add_argument(
        "train_conllu", type=str, help="Path to the training set in CoNLL-U format"
    )
    parser.add_argument(
        "test_conllu", type=str, help="Path to the test set in CoNNL-U format"
    )
    args = parser.parse_args()

    train_sents = read_conllu(args.train_conllu)
    test_sents = read_conllu(args.test_conllu)

    train_tokens = list(map(extract_tokens, train_sents))
    train_tags = extract_tags(train_sents)

    test_tokens = list(map(extract_tokens, test_sents))
    test_tags = extract_tags(test_sents)

    tagger = MostFrequentTagger(tokens=train_tokens, tags=train_tags)

    train_pred_tags = list(map(tagger.tag, train_tokens))
    test_pred_tags = list(map(tagger.tag, test_tokens))

    print("Evaluation on train set")
    pretty_metrics(evaluate_dataset(train_pred_tags, train_tags))
    print("Evaluation on test set")
    pretty_metrics(evaluate_dataset(test_pred_tags, test_tags))

    # small test
    from sklearn.preprocessing import LabelEncoder

    true_list = [item for sublist in test_tags for item in sublist]
    pred_list = [item for sublist in test_pred_tags for item in sublist]

    lbl = LabelEncoder()
    lbl.fit(true_list)

    true_list = list(map(lbl.transform, [true_list]))
    pred_list = list(map(lbl.transform, [pred_list]))

    from sklearn.metrics import precision_recall_fscore_support

    print(precision_recall_fscore_support(true_list[0], pred_list[0], average=None))
    print(precision_recall_fscore_support(true_list[0], pred_list[0], average="macro"))
    print(precision_recall_fscore_support(true_list[0], pred_list[0], average="micro"))


if __name__ == "__main__":
    main()
