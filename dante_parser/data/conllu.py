import re


def read_conllu(path: str) -> list:
    """
    Reads conllu and split sentences.

    Parameters
    ----------
    path: str
        Path to the conllu file.

    Returns
    -------
    list:
        List of sentences.
    """
    conllu_sentence_regex = r"(# [newdoc|text][\s\S]*?[\r\n]{2})"
    sents = None

    with open(path, "r") as in_file:
        data = in_file.read()
        
        sents = re.findall(conllu_sentence_regex, data)

    return sents

def write_conllu(file_name: str, sents: list):
    """
    Create conllu file with given sentences.

    Parameters
    ----------
    file_name: str
        Output filename.
    sents: list
        List of strings.
    """

    with open(file_name, "w") as out_f:
        for sent in sents:
            if sent: # Skip empty sentences.
                out_f.write(sent)

