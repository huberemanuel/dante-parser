import argparse
import os
import re

from xlrd import open_workbook

from dante_parser.data.conllu import read_conllu, extract_ids


def extract_pos_tags(input_xls: str) -> dict:
    """
    Read all post-tas from JUIZ column and returns a dictionary
    indexed by the sentence id with a list of all pos-tags.

    Parameters
    ----------
    input_xls: str
        Input xls file.

    Returns
    -------
    dict:
        Dict of sentence ids and pos-tags.
    """
    pos_tags = dict()

    with open(input_xls, "rb") as in_xls:
        wbook = open_workbook(file_contents=in_xls.read())
        
        for sheet in wbook.sheets():
            r_sent = False # Reading sentence state
            for row in range(sheet.nrows):
                if not r_sent and "dante_01" in sheet.cell(row, 0).value:
                    sent_id = sheet.cell(row, 0).value
                    pos_tags[sent_id] = []
                    r_sent = True
                else:
                    if len(sheet.cell(row,0).value) == 0:
                        r_sent = False
                    else:
                        token = sheet.cell(row, 4).value
                        if token != "JUIZ":
                            pos_tags[sent_id].append(sheet.cell(row, 4).value)

    return pos_tags

def replace_tags(conllu_path: str, pos_tags: dict, debug:bool = False) -> list:
    """
    Replace all tags from original CoNLL-U file with `pos-tags`.

    Parameters
    ----------
    conllu_path: str
        Input CoNLL-U file.
    pos_tags: dict
        Dicitonary with pos-tags.
    debug: bool
        Print pos-tags changes.

    Returns
    -------
    str:
        Processed CoNLL-U.
    """

    sent_ids = extract_ids(conllu_path)

    with open(conllu_path, "r") as in_f:
        conllu_data = in_f.read()

    conllu_data = conllu_data.split("\n")
    i_sent_id = -1
    i_token = 0
    r_sentence = False # Reading sentence state

    for i, line in enumerate(conllu_data):
        if len(line.strip()) == 0:
            r_sentence = False
            continue
        if not r_sentence and line[0] == "#":
            r_sentence = True
            i_sent_id += 1
            i_token = 0
            continue
        elif line[0] == "#": # Skip header
            continue

        if r_sentence:
            sent_id = sent_ids[i_sent_id]
            token = line.split("\t")[1]
            ori_tag = line.split("\t")[3]
            tag = pos_tags[sent_id][i_token]
            if ori_tag != tag:
                conllu_data[i] = line.replace(f"\t{ori_tag}\t", f"\t{tag}\t", 1)
                if debug:
                    print(f"Sentence {sent_id} replaced for token {token} from {ori_tag} to {tag}")
            i_token += 1

    out_conllu = ""
    for line in conllu_data:
        out_conllu += line + "\n"

    return out_conllu

def main():
    parser = argparse.ArgumentParser("Create CoNLL-U from adjucated xls input")
    parser.add_argument("input_xls", type=str, help="Input xls with adjucated sentences")
    parser.add_argument("input_conllu", type=str, help="Input conllu to add pos tags")
    parser.add_argument("out_filename", type=str, help="Output file name")
    args = parser.parse_args()

    pos_tags = extract_pos_tags(args.input_xls)
    out_conllu = replace_tags(args.input_conllu, pos_tags)

    with open(args.out_filename, "w") as out_f:
        out_f.write(out_conllu)

if __name__ == "__main__":
    main()

