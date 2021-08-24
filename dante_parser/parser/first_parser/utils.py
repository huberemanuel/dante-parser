from conllu import TokenList

from dante_parser.parser.first_parser.tree import DependencyTree


def read_tree(sentence: TokenList) -> DependencyTree:
    """
    Read a TokenList and stores the dependency relations on a
    DependencyTree strcture.

    Parameters
    ----------
    sentence: TokenList
        List of tokens read from `conllu` package.

    Returns
    -------
    DependencyTree:
        Parsed tree.
    """
    tree = DependencyTree()
    for token in sentence:
        if not isinstance(token["id"], tuple):
            tree.add(token["head"], token["deprel"])
    return tree
