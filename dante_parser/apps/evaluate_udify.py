import argparse

from dante_tokenizer.evaluate import evaluate_dataset

from dante_parser.data.conllu import extract_tags, read_conllu

def main():
    parser = argparse.ArgumentParser("Evaluate Udify model")
    parser.add_argument("pred_file", type=str, help="Path to udify output")
    parser.add_argument("true_file", type=str, help="Path to true conllu")
    args = parser.parse_args()

    if not args.pred_file.endswith(".conllu"):
        raise ValueError("pred_file must be in CoNNL-U format")

    pred_input = read_conllu(args.pred_file, no_header=True)
    true_input = read_conllu(args.true_file)

    assert len(pred_input) == len(true_input), f"pred_input: {len(pred_input)} true_input: {len(true_input)}"

    pred_tags = extract_tags(pred_input)
    true_tags = extract_tags(true_input)


    print(evaluate_dataset(pred_tags, true_tags))


if __name__ == "__main__":
    main()

