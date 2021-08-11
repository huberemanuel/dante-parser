import operator
from typing import Dict, List


class MostFrequentTagger:
    """
    Count frequency of token-tag pairs and predicts the most frequent tag for the given
    word.
    OOVs are treated as X tag.
    """

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
    UNK = "X"

    def __init__(self, tokens: List[List[str]], tags: List[List[str]]):
        """
        Creates MostFrequentTagger instance and computes the frequency counter

        Parameters
        ----------
        tokens: List[List[str]]
            List of lists of tokens.
        tags: List[List[str]]
            List of lists of tags.
        """
        if len(tokens) != len(tags):
            raise ValueError("`tokens` and `tags` must have the same length")
        for token, tag in zip(tokens, tags):
            if len(token) != len(tag):
                raise ValueError("`tokens` and `tags` must have the same length")
        self.counter = self._compute_counter(tokens, tags)

    def tag(self, tokens: List[str]) -> List[str]:
        """
        Finds tag on self.counter dictionary for all tokens.
        OOV cases are treated as X symbols.

        Parameters
        ----------
        tokens: List[str]
            List of tokens representing the input sentence.

        Returns
        -------
        List[str]
            List of predicted tags.
        """
        tags = []
        for token in tokens:
            if token not in self.counter.keys():
                tags.append("X")
            else:
                tags.append(self.counter[token])
        return tags

    @staticmethod
    def _compute_counter(
        tokens: List[List[str]], tags: List[List[str]]
    ) -> Dict[str, str]:
        """
        Count token-tag frequency and return dictionary with the most
        freq tag for a given word.

        Parameters
        ----------
        tokens: List[List[str]]
            List of lists of tokens.
        tags: List[List[str]]
            List of lists of tags.

        Returns
        -------
        Dict[str, str]:
            Token-tag dict for predictions.
        """
        tags = MostFrequentTagger._clean_tags(tags)
        unique_words = {x for sublist in tokens for x in sublist}
        counter = {w: {k: 0 for k in MostFrequentTagger.ud_tags} for w in unique_words}

        for sent_tokens, sent_tags in zip(tokens, tags):
            for token, tag in zip(sent_tokens, sent_tags):
                counter[token][tag] += 1

        most_freq = {w: None for w in unique_words}
        for key, item in counter.items():
            most_freq[key] = max(item.items(), key=operator.itemgetter(1))[0]

        return most_freq

    @staticmethod
    def _clean_tags(tags: List[List[str]]) -> List[List[str]]:
        """
        Rename tags that aren't in UD standards to "X".

        Parameters
        ----------
        tags: List[List[str]]
            List of lists of tags.

        Returns
        -------
        List[List[str]]
            Cleaned tags.
        """
        return list(
            map(
                lambda x: list(
                    map(
                        lambda y: y
                        if y in MostFrequentTagger.ud_tags
                        else MostFrequentTagger.UNK,
                        x,
                    )
                ),
                tags,
            )
        )
