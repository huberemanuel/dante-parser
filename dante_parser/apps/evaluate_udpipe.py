import argparse

from dante_tokenizer.evaluate import evaluate_dataset

from ufal.udpipe import Model, InputFormat
from dante_parser.parser.udpipe import predict_udpipe 
from dante_parser.data.conllu import read_conllu, remove_tags, extract_tags

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
    pred_tags = extract_tags(preds)
    true_tags = extract_tags(true)

    print(evaluate_dataset(pred_tags, true_tags))

if __name__ == "__main__":
    main()

