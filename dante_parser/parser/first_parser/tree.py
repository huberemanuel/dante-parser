from enum import Enum


class Node(Enum):
    NONEXIST = -1
    ROOT = 0
    UNKOWN = 1
    NULL = 2


class DependencyTree:
    """
    Represents a dependency relation tree.
    """

    def __init__(self) -> None:
        self.n = 0
        self.head = [Node.NONEXIST.name]
        self.label = [Node.UNKOWN.name]
        self.counter = -1

    def add(self, head: str, label: str) -> None:
        """
        Add a new token to the Tree.

        Parameters
        ----------
        head: str
            Head of the new token.
        label: str
            Dependency relation of the given token and its head.
        """
        self.n += 1
        self.head.append(head)
        self.label.append(label)

    def set_relation(self, dependent: int, head: int, label: str) -> None:
        self.head[dependent] = head
        self.label[dependent] = label

    def set_dependency(self, token_idx: int, head_idx: int, label: str) -> None:
        """
        Create a relation between the token from `token_idx` and head from `head_idx`
        with the given dependency relation label.
        Obs: This function does not verify if the `label` is a valid one.

        Parameters
        ----------
        token_idx: int
            Index of the dependent node.
        head_idx: int
            Index of the head node.
        label: str
            Label of the dependency.
        """
        self.head[token_idx] = head_idx
        self.label[token_idx] = label

    def get_head(self, index: int) -> int:
        """
        Get head from the given node index
        """
        if index <= 0 or index > self.n:
            return Node.NONEXIST.value
        return self.head[index]

    def get_label(self, index: int) -> int:
        """
        Get the label from the given node index
        """
        if index <= 0 or index > self.n:
            # Certeza? No índice zero não deveria retornar o ROOT?
            return Node.NULL.name
        return self.label[index]

    def get_root(self) -> int:
        """
        Get the index of the root node.
        """
        root = 0
        for i in range(1, self.n + 1):
            if self.get_head(i) == root:
                return i
        return 0

    def is_single_root(self) -> bool:
        """
        Check if the tree has more than one root.
        """
        roots = 0
        root = 0
        for i in range(1, self.n + 1):
            if self.get_head(i) == root:
                roots += 1
        return roots == 1

    def is_valid_tree(self) -> bool:
        """
        Check if the tree is a valid dep tree.
        If any dependent has a head that is not in the range of the
        stored nodes (0 ~ n), then it is invalid.
        Walks from all nodes to root, if a walk is not possible,
        then it is not a valid tree.
        """
        visiteds = []
        visiteds.append(-1)
        for i in range(1, self.n + 1):
            if self.get_head(i) < 0 or self.get_head(i) > self.n:
                return False
            visiteds.append(-1)

        for i in range(1, self.n + 1):
            k = i
            while k > 0:
                if visiteds[k] >= 0 and visiteds[k] < i:
                    break
                if visiteds[k] == i:
                    return False
                visiteds[k] = i
                k = self.get_head(k)

        return True

    def is_projective(self) -> bool:
        """
        Check if the tree is projective.
        Iterates from left to right on the sentence relations.
        Check all elements on the left to the actual word.
        If one of them is the depedent of the actual, clal the function
        recursively.
        After all elements on the left are processed, we check if the
        counter is equal to number of tokens from left to the actual position
        if it is different, then a dependent to the left wasn't processed
        and it is child from a non-projective relation.
        The same procedure is applied to the right of the current token.
        """
        if not self.is_valid_tree():
            return False
        self.counter = -1
        return self.visit_tree(0)

    def visit_tree(self, w) -> bool:
        """
        Inner recursive function for checking projectivity of subtree.
        """
        for i in range(1, w):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        self.counter += 1
        if w != self.counter:
            return False
        for i in range(w + 1, self.n + 1):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        return True

    def equal(self, t) -> bool:
        """
        Check if self and t are equal, asserting number of elements,
        heads and labels.
        """
        if t.n != self.n:
            return False
        for i in range(1, self.n + 1):
            if self.get_head(i) != t.get_head(i):
                return False
            if self.get_label(i) != t.get_label(i):
                return False
        return True

    def print_tree(self) -> None:
        for i in range(1, self.n + 1):
            print(str(i) + " " + str(self.get_head(i)) + " " + self.get_label(i))
        print("\n")
