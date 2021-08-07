import argparse
import os

import xlwt

from dante_parser.data import extract_ids, extract_tags, extract_tokens, read_conllu


def main():
    parser = argparse.ArgumentParser("Converts CoNNL-U to xls")
    parser.add_argument("input_conllu", type=str, help="Path to input conllu file.")
    parser.add_argument("--out_filename", type=str, help="Output name")
    args = parser.parse_args()

    sent_texts = read_conllu(args.input_conllu)
    sent_ids = extract_ids(args.input_conllu)
    sent_tags = extract_tags(sent_texts)
    sent_tokens = list(map(extract_tokens, sent_texts))

    id_style = xlwt.easyxf(
        "font: name Arial, bold on, color white;"
        "pattern: pattern solid, pattern_fore_colour red, pattern_back_colour red;"
        "borders: left thin, right thin, top thin, bottom thin;"
    )
    header_style = xlwt.easyxf(
        "font: name Arial, color white;"
        "pattern: pattern solid, pattern_fore_colour blue, pattern_back_colour blue;"
        "borders: left thin, right thin, top thin, bottom thin;"
    )
    tag_style = xlwt.easyxf(
        "font: name Arial, bold on;"
        "pattern: pattern solid, pattern_fore_colour yellow, pattern_back_colour yellow;"
        "borders: left thin, right thin, top thin, bottom thin;"
    )
    token_style = xlwt.easyxf(
        "font: name Arial; borders: left thin, right thin, top thin, bottom thin;"
    )

    assert (
        len(sent_ids) == len(sent_tags) == len(sent_tokens) == len(sent_texts)
    ), "{} - {} - {} - {}".format(
        len(sent_ids), len(sent_tags), len(sent_tokens), len(sent_texts)
    )

    book = xlwt.Workbook()
    sheet = book.add_sheet("JUIZ", cell_overwrite_ok=True)
    n_row = 0

    for sent_id, sent_token, sent_tag in zip(sent_ids, sent_tokens, sent_tags):
        row = sheet.row(n_row)

        assert len(sent_token) == len(sent_tag)

        row.write(0, sent_id, id_style)
        row.write(1, "", id_style)

        n_row += 1
        row = sheet.row(n_row)
        for i, text in enumerate(["token", "JUIZ"]):
            row.write(i, text, header_style)

        for i, (token, tag) in enumerate(zip(sent_token, sent_tag)):
            n_row += 1
            row = sheet.row(n_row)
            row.write(0, token, token_style)
            row.write(1, tag, tag_style)
        n_row += 2

    out_name = (
        args.out_filename or os.path.basename(args.input_conllu).split(".")[0] + ".xlsx"
    )
    book.save(out_name)


if __name__ == "__main__":
    main()
