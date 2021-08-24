import pickle
from collections import Counter
from typing import List

from dante_parser.parser.first_parser.tree import DependencyTree, Node


class Vocabulary:
    def __init__(self, sentences, trees: List[DependencyTree]) -> None:
        self.word_token_to_id = {}
        self.pos_token_to_id = {}
        self.label_token_to_id = {}
        self.id_to_token = {}

        word = []
        pos = []
        label = []
        for sentence in sentences:
            for token in sentence:
                if not isinstance(token["id"], tuple):
                    word.append(token["form"])
                    pos.append(token["upos"])

        root_label = None
        for tree in trees:
            for k in range(1, tree.n + 1):
                if tree.get_head(k) == 0:
                    root_label = tree.get_label(k)
                else:
                    label.append(tree.get_label(k))

        # Por que remover a root label das labels?
        if root_label in label:
            label.remove(root_label)

        index = 0
        word_count = [Node.UNKOWN.name, Node.NULL.name, Node.ROOT.name]
        # The extend method will only add the keys from the Counter.
        word_count.extend(Counter(word))
        for word in word_count:
            self.word_token_to_id[word] = index
            self.id_to_token[index] = word
            index += 1

        pos_count = [Node.UNKOWN.name, Node.NULL.name, Node.ROOT.name]
        pos_count.extend(Counter(pos))
        for pos in pos_count:
            self.pos_token_to_id[pos] = index
            self.id_to_token[index] = pos
            index += 1

        label_count = [Node.NULL.name, root_label, Node.UNKOWN.name]
        label_count.extend(Counter(label))
        for label in label_count:
            self.label_token_to_id[label] = index
            self.id_to_token[index] = label
            index += 1

    def size(self) -> int:
        return (
            len(self.word_token_to_id)
            + len(self.pos_token_to_id)
            + len(self.label_token_to_id)
        )

    def get_word_id(self, token: str) -> int:
        if token in self.word_token_to_id:
            return self.word_token_to_id[token]
        return self.word_token_to_id[Node.UNKOWN.name]

    def get_pos_id(self, token: str):
        if token in self.pos_token_to_id:
            return self.pos_token_to_id[token]
        return self.pos_token_to_id[Node.UNKOWN.name]

    def get_label_id(self, token: str):
        if token in self.label_token_to_id:
            return self.label_token_to_id[token]
        return self.label_token_to_id[Node.UNKOWN.name]

    def save(self, pickle_file_path: str) -> None:
        with open(pickle_file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, pickle_file_path: str) -> "Vocabulary":
        with open(pickle_file_path, "rb") as file:
            vocabulary = pickle.load(file)
        return vocabulary
