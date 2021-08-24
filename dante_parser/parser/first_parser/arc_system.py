from typing import List

from dante_parser.parser.first_parser.tree import DependencyTree, Node


class ArcSystem:
    """
    Basic arc system for projective arc system.
    It does not handle non-projective sentences or multi-head trees.
    """

    def __init__(self, tokens: List[str]) -> None:
        self.stack = []
        length = len(list(filter(lambda x: isinstance(x["id"], int), tokens)))
        self.buffer = [Node.ROOT.value] + list(range(1, length + 1))
        self.tree = DependencyTree()
        for i in range(1, length + 1):
            self.tree.add(Node.NONEXIST.value, Node.UNKOWN.name)
        self.sentence = tokens

    def get_word(self, index: int) -> str:
        """
        Get word at given index.

        Parameters
        ----------
        index: int
            Index of token.

        Returns
        -------
        str:
            Word form for given node at `index`.
        """
        if index == 0:
            return Node.ROOT.name
        else:
            index -= 1

        if index < -0 or index >= len(self.sentence):
            return Node.NULL.name
        else:
            return self.sentence[index]["form"]

    def get_pos(self, index: int) -> str:
        """
        Get upos at given index.

        Parameters
        ----------
        index: int
            Index of token.

        Returns
        -------
        str:
            UPOS tag for given node at `index`.
        """
        if index == 0:
            return Node.ROOT.name
        else:
            index -= 1

        if index < -0 or index >= len(self.sentence):
            return Node.NULL.name
        else:
            return self.sentence[index]["upos"]

    def get_label(self, index: int) -> str:
        """
        Get dependency relation label at given index.

        Parameters
        ----------
        index: int
            Index of token.

        Returns
        -------
        str:
            Dependency relation between index and its head.
        """
        return self.tree.get_label(index)

    def shift(self) -> bool:
        """Apply shift operation if possible."""
        if len(self.buffer) < 1:
            return False  # Could not make a shift.
        else:
            element = self.buffer.pop(0)
            self.stack.append(element)
            return True

    def is_empty(self) -> bool:
        """Verify if both buffer and stack are empty or not."""
        return len(self.buffer) == 0 and len(self.stack) <= 1

    def has_other_child(self, word: int, gold_tree: DependencyTree) -> bool:
        """
        Check if the gold tree has a relation that the current tree hasn't.

        Parameters
        ----------
        word: int
            Index of the anchor word.
        gold_tree: DependencyTree
            Tree for the treebank.

        Returns
        -------
        bool:
            Whether exists another child that has the same parent of `word` or not.
        """
        for i in range(1, gold_tree.n + 1):
            if i > self.tree.n and gold_tree.get_head(i) == word:
                return True
            elif gold_tree.get_head(i) == word and self.tree.get_head(i) != word:
                return True
        return False

    def get_stack(self, index: int) -> int:
        """
        Get token at `index` if exists.

        Parameters
        ----------
        index: int
            Index of token.

        Returns
        -------
        int:
            Token on the stack at given index.
        """
        if index == 0 and len(self.stack) >= 1:
            return self.stack[-1]

        n = index + 1
        if index == 0 or n > len(self.stack):
            return Node.NONEXIST.value

        return self.stack[-n]

    def get_buffer(self, index: int) -> int:
        """
        Get token at `index` if exists.

        Parameters
        ----------
        index: int
            Index of token.

        Returns
        -------
        int:
            Token on the buffer at given index.
        """
        if index < 0 or index >= len(self.buffer):
            return Node.NONEXIST.value

        return self.buffer[index]

    def apply(self, transition: str) -> bool:
        """
        Apply given transition on actual configuration.

        Parameters
        ----------
        transition: str
            Valid transition for the actual ArcSystem.

        Returns
        -------
        bool:
            Whether transition was applied successfully or not.
        """
        if not self.can_apply(transition):
            return False

        first_word = self.get_stack(0)
        second_word = self.get_stack(1)
        label = transition[2:]

        if transition.startswith("R"):
            self.tree.set_dependency(first_word, second_word, label)
            self.stack.pop(-1)
        elif transition.startswith("L"):
            self.tree.set_dependency(second_word, first_word, label)
            self.stack.pop(-2)
        else:
            self.shift()

        return True

    def can_apply(self, transition: str) -> bool:
        """
        Check if it's possible to apply `transition` on actual configuration

        Parameters
        ----------
        transition: str
            Valid transition on the ArcSystem

        Returns
        -------
        bool:
            Whether it is possible to apply transition on actual configuration.
        """
        if transition.startswith("R") or transition.startswith("L"):
            label = transition[2:]
            if transition.startswith("L"):
                head = self.get_stack(0)
            else:
                head = self.get_stack(1)

            if head < 0:
                return False
            if head == 0 and label != "root":
                return False

        n_stack = len(self.stack)
        n_buffer = len(self.buffer)

        if transition.startswith("L"):
            # Meu pitaco, posso sim se o tamanho for igual a 2
            return n_stack >= 2
        elif transition.startswith("R"):
            return (n_stack > 2) or (n_stack == 2 and n_buffer == 0)
        # Push operation
        return n_buffer > 0

    def get_left_child(self, index: int, count: int) -> int:
        """
        Get cnt-th leftmost child of index.
        (i.e., if count = 1, the leftmost child of index will be returned,
                if count = 2, the 2nd leftmost child of index will be returned.)
        Obs: Used for extracting features for models.
        """
        if index < 0 or index > self.tree.n:
            return Node.NONEXIST.value

        c = 0
        for i in range(1, index):
            if self.tree.get_head(i) == index:
                c += 1
                if c == count:
                    return i

        return Node.NONEXIST.value

    def get_right_child(self, index: int, count: int) -> int:
        """
        Get cnt-th leftmost child of index.
        (i.e., if count = 1, the rightmost child of index will be returned,
                if count = 2, the 2nd leftmost child of index will be returned.)
        Obs: Used for extracting features for models.
        """
        if index < 0 or index > self.tree.n:
            return Node.NONEXIST.value

        c = 0
        for i in range(self.tree.n, index, -1):
            if self.tree.get_head(i) == index:
                c += 1
                if c == count:
                    return i

        return Node.NONEXIST.value

    @staticmethod
    def get_oracle_decision(configuration: "ArcSystem", tree: DependencyTree) -> str:
        """
        Given an ArcSystem `configuration` and the gold `tree`, gets the next
        transition that the system should make.

        Parameters
        ----------
        configuration: ArcSystem
            Actual configuration of an ArcSystem.
        tree: DependencyTree
            Gold tree from treebank.

        Returns
        -------
        str:
            Generated transition.
        """
        first_word = configuration.get_stack(0)
        second_word = configuration.get_stack(1)
        if (
            first_word == Node.NONEXIST.value
            or second_word == Node.NONEXIST.value
            and len(configuration.buffer) == 0
        ):
            return "P"
        if tree.get_head(second_word) == first_word:
            return "L-{}".format(tree.get_label(second_word))
        elif tree.get_head(
            first_word
        ) == second_word and not configuration.has_other_child(first_word, tree):
            return "R-{}".format(tree.get_label(first_word))

        return "P"

    @staticmethod
    def generate_transitions(relations: List[str]) -> List[str]:
        """
        Generate all possible transitions given the dependency `relations`.

        Parameters
        ----------
        relations: List[str]
            Possible dependency relations.

        Returns
        -------
        List[str]
            All possible transitions.
        """
        # TODO: Automatically reads relations from UD standards
        transitions = []

        for rel in relations:
            transitions.append("R-{}".format(rel))
            transitions.append("L-{}".format(rel))
        transitions.append("P")

        return transitions
