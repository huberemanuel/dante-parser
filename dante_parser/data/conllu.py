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

