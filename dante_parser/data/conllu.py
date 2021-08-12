import re


def extract_tokens(sentence: str) -> list:
    """
    Extract tokens from input sentence conllu.

    Parameters
    ----------
    sentence: str
        Sentence on CoNLL-U format

    Retunrs
    -------
    list:
        List of tokens.
    """

    conllu_tokens_regex = r"^\d+(?:\-\d+)?\t([^\t]*)"
    tokens = re.findall(conllu_tokens_regex, sentence, re.MULTILINE)

    return tokens


def extract_ids(path: str) -> list:
    """
    Read a list of sentences on CoNLL-U format and return the list
    of sentence's ids.

    Parameters
    ----------
    path: str
        Path to input CoNLL-U file.

    Returns
    -------
    list:
        List of ids.
    """

    ids = []
    conllu_sentence_id_regex = r"sent_id = (dante_01_.*)"

    with open(path, "r") as conllu_file:
        conllu_data = conllu_file.read()
        ids = re.findall(conllu_sentence_id_regex, conllu_data)

    return ids


def read_conllu(path: str, no_header=False) -> list:
    """
    Reads conllu and split sentences.

    Parameters
    ----------
    path: str
        Path to the conllu file.
    no_header: bool
        Used when sentences have no header.

    Returns
    -------
    list:
        List of sentences.
    """
    conllu_sentence_regex = r"(# [newdoc|text][\s\S]*?[\r\n]{2})"
    sents = None

    with open(path, "r") as in_file:
        data = in_file.read()

        if no_header:
            return data.split("\n\n")
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
            if sent:  # Skip empty sentences.
                out_f.write(sent)


def remove_tags(sent: str) -> str:
    """
    Replace every tag with "_".

    Parameters
    ----------
    sent: str
        Input string on CoNLL-U format.

    Returns
    -------
    str:
        Processed string.
    """
    return re.sub(
        r"(^\d+(?:\-\d+)?\t*(?:[^\t]*)\t(?:[^\t]*)\t)(\w+)",
        r"\1_",
        sent,
        flags=re.MULTILINE,
    )


def extract_tags(sents: list) -> list:
    """
    Returns tags from all sentences.

    Parameters
    ----------
    sents: list
        List of str on CoNLL-U format.

    Returns
    -------
        List of str with respective pos-tags.
    """

    return list(
        map(
            lambda x: re.findall(
                r"^\d+(?:\-\d+)?\t*(?:[^\t]*)\t(?:[^\t]*)\t([^\t]*)",
                x,
                flags=re.MULTILINE,
            ),
            sents,
        )
    )


def remove_multiword_rows(sents: list) -> list:
    """
    Remove multiword rows from input CoNLL-U sentences

    Parameters
    ----------
    sents: list
        List of str on CoNLL-U format.

    Returns
    -------
        List of str with respective pos-tags.
    """

    return list(
        map(
            lambda x: re.sub(
                r"^\d+(?:\-\d+).*$",
                "",
                x,
                flags=re.MULTILINE,
            ),
            sents,
        )
    )
