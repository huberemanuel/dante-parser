from typing import List
import spacy


class SimpleParser:
    def __init__(self) -> None:
        self.nlp = spacy.load("pt_core_news_sm")

    def sentence_segmentation(self, sentence: str) -> List[str]:
        """
        Uses Statistical sentence segmenter from Spacy.

        Parameters
        ----------
        sentences: List[str] or str
            Input text to be segmented.

        Returns
        -------
        List[str]
            List of segmenteted sentences
        """
        self.nlp.enable_pipe("senter")

        if not isinstance(sentence, str):
            raise ValueError("Input text is not instance of str ({})".format(sentence))

        return self.nlp(sentence).sents

    def tokenize(self, sentence: str) -> List[str]:
        """
        Uses default spacy tokenization.

        Parameters
        ----------
        sentence: str
            Input sentence.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(sentence, str):
            raise ValueError("Input text is not instance of str ({})".format(sentence))

        return map(lambda x: x.text, self.nlp(sentence))

    def tag(self, tokens: List[str]) -> List[str]:
        """
        Returning pos-tags.

        Parameters
        ----------
        tokens: List[str]
            Input sentence tokens.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(tokens, list):
            raise ValueError(
                "Input text is not instance of List[str] ({})".format(tokens)
            )

        tokens = " ".join(tokens)
        return map(lambda x: x.pos_, self.nlp(tokens))

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Returning lemmas.

        Parameters
        ----------
        tokens: List[str]
            Input sentence tokens.
        Returns
        -------
        List[str]:
            List of tokenized symbols.
        """

        if not isinstance(tokens, list):
            raise ValueError(
                "Input text is not instance of List[str] ({})".format(tokens)
            )

        tokens = " ".join(tokens)
        return map(lambda x: x.lemma_, self.nlp(tokens))
