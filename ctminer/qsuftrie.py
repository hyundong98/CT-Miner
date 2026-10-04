from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Protocol, Sequence, Final


INF: Final = 999999999


def label_inf(char: str | int) -> str:
    """
    Just a helper for labelling
    """
    if char == INF:
        return "∞"
    else:
        return str(char)


def label_int_zero(char: int) -> int:
    """
    Another helper, but only works with int string and embeds inf to 0.
    """
    if char == INF:
        return 0
    else:
        return char


def sort_nodes(nodes: Sequence["Node"]):
    """
    Sort by... specific order.
    (Dec. Length -> Dec. Leaves)
    """
    n = len(nodes)

    # # sort by index
    # idx = [[] for _ in range(n)]

    # for nd in nodes:
    #     idx[nd.index].append(nd)

    # nodes = [nd for bucket in idx for nd in bucket]

    # sort by leaf count (DECREASING)
    lfs = [[] for _ in range(n + 1)]

    for nd in nodes:
        lfs[nd.leaf_count].append(nd)

    nodes = [nd for l in range(n, -1, -1) for nd in lfs[l]]

    # sort by length (DECREASING)
    lns = [[] for _ in range(n + 1)]

    for nd in nodes:
        lns[nd.length].append(nd)

    nodes = [nd for l in range(n, -1, -1) for nd in lns[l]]

    return nodes


class CharacterOracle(Protocol):
    """
    The Cole-Hariharan algorithm assumes that the j-th character
    of s_i can be obtained in O(1) time.
    """

    def char(self, string_id: int, position: int) -> str: ...

    def length(self, string_id: int) -> int: ...


class StringOracle:
    """
    Simple oracle backed by ordinary Python strings.
    Left for testing purpose
    """

    def __init__(self, strings: Sequence[str]):
        self.strings = list(strings)

    def char(self, idx: int, position: int) -> str:
        return self.strings[idx][position]

    def length(self, idx: int) -> int:
        return len(self.strings[idx])


class PDOracle:
    """
    Oracle suited for PD representation of integer string.

    Input should be a valid PDRep with an endmarker.
    """

    def __init__(self, chars: Sequence[int]):
        self.chars = list(chars)

    def char(self, idx: int, position: int) -> int:
        if self.chars[idx + position] <= position:
            return self.chars[idx + position]
        else:
            return INF

    def length(self, idx: int) -> int:
        return len(self.chars) - idx


class NodeType(Enum):
    REAL = auto()
    IMAGINARY = auto()
    BACK_PROPAGATED = auto()
    LEAF = auto()


@dataclass
class Node:
    """
    A node in the compacted trie.
    """

    parent: Optional["Node"]

    source: int
    start: int
    end: int
    depth: int
    node_type: NodeType

    leaf_count: int = 0
    string_index: int = -1
    children: dict[str | int, "Node"] = field(default_factory=dict)
    suffix_link: Optional["Node"] = None

    # Extra property to use for various alg.
    mark: bool = False

    @property
    def edge_length(self) -> int:
        return self.end - self.start

    @property
    def length(self) -> int:
        return self.depth

    @property
    def index(self) -> int:
        return self.source

    @property
    def leaves(self) -> int:
        return self.leaf_count

    @property
    def is_leaf(self) -> bool:
        return self.node_type is NodeType.LEAF

    @property
    def is_real(self) -> bool:
        return self.node_type is NodeType.REAL


@dataclass(frozen=True)
class Locus:
    """
    A position in the compacted trie.
    """

    node: Node
    offset: int

    @property
    def depth(self) -> int:
        parent = self.node.parent
        if parent is None:
            parent_depth = 0
        else:
            parent_depth = parent.depth

        return parent_depth + self.offset

    @property
    def is_explicit(self) -> bool:
        return self.offset == self.node.edge_length


class CHSuffixTree:
    """
    Compacted trie construction for a quasi-suffix collection,
    following the Cole-Hariharan modification of McCreight's
    algorithm.

    Input strings are indirectly given by the O(1) oracle.

    The Python implementation uses dictionaries for child lookup.
    Python dictionaries give expected O(1) child lookup.

    n strictly includes the endmarker.
    """

    def __init__(self, oracle: CharacterOracle, n: int):
        if n <= 0:
            raise RuntimeError(f"__init__(): n must be positive (got {n})")

        self.oracle = oracle
        self.n = n

        self._validate_lengths()

        self.root = Node(
            parent=None, source=-1, start=0, end=0, depth=0, node_type=NodeType.REAL
        )

        # Empty string links to itself.
        self.root.suffix_link = self.root

        self.leaves: list[Optional[Node]] = [None for _ in range(n)]

    def char(self, idx: int, position: int) -> str | int:
        return self.oracle.char(idx, position)

    def length(self, idx: int) -> int:
        return self.oracle.length(idx)

    def _validate_lengths(self) -> None:
        for i in range(self.n):
            expected = self.n - i
            actual = self.length(i)

            if actual != expected:
                raise RuntimeError(
                    f"_validate_lengths(): Invalid value of |s[{i}]| (got: {actual}, expected: {expected})"
                )

    def edge_label(self, node: Node) -> str:
        return " ".join(
            label_inf(self.char(node.source, p)) for p in range(node.start, node.end)
        )

    # def path_label(self, node: Node) -> str:
    #     """
    #     Legacy implementation
    #     """
    #     pieces = []

    #     while node is not self.root:
    #         pieces.append(self.edge_label(node))
    #         node = node.parent
    #     pieces.reverse()

    #     return " ".join(pieces)

    def path_label(self, node: Node) -> str:
        return " ".join(label_inf(self.char(node.source, p)) for p in range(node.depth))

    def path_label_truncate(
        self, node: Node, length: int, return_as_list: bool = False
    ) -> str | Sequence[int]:
        """
        If return_as_list is set to True, then inf is returned as 0.
        """
        if length > node.depth:
            raise RuntimeError(
                f"path_label_truncate(): Invalid length (depth={node.depth}, got {length})"
            )

        if not return_as_list:
            return " ".join(label_inf(self.char(node.source, p)) for p in range(length))
        else:
            return [label_int_zero(self.char(node.source, p)) for p in range(length)]

    def _split_edge(self, locus: Locus, node_type: NodeType) -> Node:
        """
        Split an implicit locus.
        """

        if locus.is_explicit:
            raise RuntimeError("_split_edge(): Cannot split an explicit locus.")

        child = locus.node
        offset = locus.offset

        if not (0 < offset < child.edge_length):
            raise RuntimeError(
                f"_split_edge(): Invalid split offset (got {offset}; req 0 < off < {child.edge_length})"
            )

        parent = child.parent

        if parent is None:
            raise RuntimeError("_split_edge(): Missing parent")

        middle = Node(
            parent=parent,
            source=child.source,
            start=child.start,
            end=child.start + offset,
            depth=parent.depth + offset,
            node_type=node_type,
        )

        first_char = self.char(child.source, child.start)
        parent.children[first_char] = middle
        child.parent = middle
        child.start += offset

        middle.children[self.char(child.source, child.start)] = child

        return middle

    def _make_explicit(self, locus: Locus, node_type: NodeType) -> Node:
        if locus.is_explicit:
            return locus.node
        else:
            return self._split_edge(locus, node_type)

    def _nanc(self, x: Node) -> Node:
        """
        Find the nearest ancestor of x having a suffix link.

        x itself is included.

        If none exists, return root.
        """

        current = x

        while current is not None:
            if current.suffix_link is not None:
                return current

            current = current.parent

        return self.root

    def _path_edges(self, ancestor: Node, descendant: Node) -> list[Node]:
        """
        Return the compacted-trie edges in top-down order.
        """

        edges = []
        cur = descendant
        while cur is not ancestor:
            edges.append(cur)
            cur = cur.parent

        edges.reverse()
        return edges

    def _locus_on_path(self, ancestor: Node, descendant: Node, depth: int) -> Locus:
        """
        Find the locus at a given depth on a known tree path.
        """

        if not (ancestor.depth <= depth <= descendant.depth):
            raise RuntimeError("_locus_on_path(): Requested depth is outside path")

        if depth == ancestor.depth:
            return Locus(ancestor, ancestor.edge_length)

        cur = ancestor
        while cur is not descendant:
            next_node = None
            child = descendant

            while child.parent is not cur and child.parent is not None:
                child = child.parent

            if child.parent is cur:
                next_node = child

            if next_node is None:
                raise RuntimeError("_locus_on_path(): next_node is missing")

            if depth <= next_node.depth:
                return Locus(next_node, depth - cur.depth)

            cur = next_node

        return Locus(descendant, descendant.edge_length)

    def _scan(
        self, start: Node, source: int, start_position: int, length: int
    ) -> Locus:
        """
        Ordinary McCreight scanning.

        If the requested string ends or differs inside an edge,
        an implicit locus is returned.

        If there is no outgoing edge for the next character,
        the insertion locus is the current explicit node.
        """

        node = start
        position = start_position
        remaining = length

        while remaining > 0:
            first_char = self.char(source, position)
            child = node.children.get(first_char)

            if child is None:
                return Locus(node, node.edge_length)

            compare_length = min(remaining, child.edge_length)

            matched = 0
            while matched < compare_length:  # test for maximal match
                a = self.char(source, position + matched)
                b = self.char(child.source, child.start + matched)

                if a != b:
                    break

                matched += 1

            if matched < compare_length:
                return Locus(child, matched)

            if remaining < child.edge_length:
                return Locus(child, remaining)

            position += child.edge_length
            remaining -= child.edge_length
            node = child

        return Locus(node, node.edge_length)

    def _rescan(
        self, start: Node, source: int, start_position: int, length: int
    ) -> tuple[Locus, list[Node]]:
        """
        Cole-Hariharan RESCAN.

        Unlike scanning, we do not compare all characters of an edge.
        Instead, only the first character of each edge is needed.

        Returns:
            final locus
            explicit nodes encountered while rescanning
        """

        node = start
        position = start_position
        remaining = length

        encountered: list[Node] = []
        ret_locus = None

        while remaining > 0:
            first_char = self.char(source, position)
            child = node.children.get(first_char)

            if child is None:  # This should not happen this time
                raise RuntimeError("_rescan(): child does not exist")

            edge_length = child.edge_length
            if edge_length <= remaining:
                position += edge_length
                remaining -= edge_length
                node = child
                encountered.append(node)
            else:
                ret_locus = Locus(child, remaining)
                return (ret_locus, encountered)

        ret_locus = Locus(node, node.edge_length)
        return (ret_locus, encountered)

    def _establish_link(self, string_index: int, x: Node) -> tuple[Node, bool]:
        """
        Returns:
            (link(x), was_existing)
        `was_existing`: True only if we used existing node. False if we had to create a node.
        """

        if x is self.root:
            return self.root, True

        nanc = self._nanc(x)

        if nanc.suffix_link is None:
            raise RuntimeError("_establish_link(): nanc has no suffix link")

        start = nanc.suffix_link
        target_depth = x.depth - 1
        start_position = max(0, nanc.depth - 1)
        start_depth = start_position
        rescan_length = target_depth - start_depth

        if rescan_length < 0:
            raise RuntimeError(
                f"_establish_link(): Invalid rescan_length (got {rescan_length})"
            )

        locus, encountered = self._rescan(
            start, string_index, start_position, rescan_length
        )

        was_existing = locus.is_explicit

        if was_existing:
            link_x = locus.node
        else:
            link_x = self._make_explicit(locus, NodeType.IMAGINARY)

        self._back_propagate(nanc, x, encountered)

        x.suffix_link = link_x

        return link_x, was_existing

    def _back_propagate(self, nanc: Node, x: Node, encountered: list[Node]) -> None:
        """
        Back-propagate every second encountered node.

        The first and last encountered nodes are excluded.
        """

        # Need at least three encountered nodes
        if len(encountered) < 3:
            return

        # Exclude head/tails and check for even-th nodes
        for c in encountered[1:-1:2]:
            target_depth = c.depth + 1

            if target_depth >= x.depth:
                continue

            locus = self._locus_on_path(nanc, x, target_depth)

            if locus.is_explicit:
                b = locus.node
                if b.suffix_link is None:
                    b.suffix_link = c
            else:
                b = self._make_explicit(locus, NodeType.BACK_PROPAGATED)
                b.suffix_link = c

    def _add_leaf(self, parent: Node, string_index: int) -> Node:
        """
        It... adds a leaf
        """

        depth = parent.depth
        length = self.length(string_index)

        if depth >= length:
            raise RuntimeError(
                f"_add_leaf(): Invalid edge detected (depth: {depth}, length: {length})"
            )

        first_char = self.char(string_index, depth)

        if first_char in parent.children:
            raise RuntimeError(
                f"_add_leaf(): The edge with given char (c: {first_char}) already exists"
            )

        leaf = Node(
            parent=parent,
            source=string_index,
            start=depth,
            end=length,
            depth=length,
            node_type=NodeType.LEAF,
            string_index=string_index,
        )

        parent.children[first_char] = leaf
        self.leaves[string_index] = leaf

        return leaf

    def preprocess_leaf_counts(self):
        def dfs(node):
            if node.is_leaf:
                node.leaf_count = 1
                return 1

            count = 0

            for child in node.children.values():
                count += dfs(child)

            if node.node_type is not NodeType.LEAF:
                node.leaf_count = count

            return count

        dfs(self.root)

    ##############################
    # Main Construction Function #
    ##############################

    def build(self):
        """
        Construct the compacted trie.
        """

        # s_1
        previous_leaf = self._add_leaf(self.root, 0)

        # s_2 ... s_n
        for i in range(1, self.n):
            x = previous_leaf.parent

            if x is None:
                raise RuntimeError("build(): previous_leaf has no parent")

            link_x, _ = self._establish_link(i, x)

            # scan from link(x) even when link(x) was newly created as an imaginary node
            remaining = self.length(i) - link_x.depth

            if remaining < 0:
                raise RuntimeError(
                    f"build(): Invalid remaining value (got {remaining})"
                )

            locus = self._scan(link_x, i, link_x.depth, remaining)

            if locus.is_explicit:
                insertion_node = locus.node
            else:
                insertion_node = self._make_explicit(locus, NodeType.REAL)

            previous_leaf = self._add_leaf(insertion_node, i)

            if (
                insertion_node.node_type
                in (NodeType.IMAGINARY, NodeType.BACK_PROPAGATED)
                and len(insertion_node.children) >= 2
            ):
                insertion_node.node_type = NodeType.REAL

        self.preprocess_leaf_counts()

        return self

    def get_leaf_count(self, node: Node) -> int:
        if node.node_type is not NodeType.REAL:
            raise RuntimeError("get_leaf_count(): Invalid leaf")
        return node.leaf_count

    def get_substr_length(self, node: Node) -> int:
        if node.node_type is not NodeType.REAL:
            raise RuntimeError("get_substr_length(): Invalid leaf")
        return node.depth

    # Debugging functions from now on
    def materialize_string(self, string_index: int) -> str:
        return " ".join(
            label_inf(self.char(string_index, p))
            for p in range(self.length(string_index))
        )

    def validate_leaves(self) -> None:
        """
        Verify that every leaf has exactly the corresponding
        input string as its root-to-leaf label.
        """

        actual = [None] * self.n

        def dfs(node: Node):
            if node.is_leaf:
                actual[node.string_index] = self.path_label(node)

            for child in node.children.values():
                dfs(child)

        dfs(self.root)

        expected = [self.materialize_string(i) for i in range(self.n)]

        if actual != expected:
            raise AssertionError(
                "validate_leaves(): Test failed\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

    def dump(self) -> None:
        def dfs(node: Node, indent: str):
            for child in node.children.values():
                label = self.edge_label(child)

                el = child.edge_length
                ln = child.depth
                lf = child.leaf_count
                idx = child.source

                if child.is_leaf:
                    kind = f"leaf s{child.string_index + 1}"
                else:
                    kind = child.node_type.name.lower()

                if child.suffix_link is None:
                    link = ""
                else:
                    link = " -> " + repr(self.path_label(child.suffix_link))

                print(
                    f"{indent}{label!r} [{kind}{link}] (ln:{ln}/lf:{lf}/idx:{idx}/el:{el})"
                )

                dfs(child, indent + "    ")

        dfs(self.root, "")

    def get_nodes(self) -> list[Node]:
        result = []

        def dfs(node: Node):
            # if node.node_type in (NodeType.REAL, NodeType.LEAF):
            #     result.append(node)
            result.append(node)

            for child in node.children.values():
                dfs(child)

        dfs(self.root)

        return result

    def set_mark_of_all_nodes(self, mark: bool) -> None:
        def dfs(node: Node):
            node.mark = mark

            for child in node.children.values():
                dfs(child, mark)

        dfs(self.root, mark)


if __name__ == "__main__":
    # strings = [
    #     "banana$",
    #     "anana$",
    #     "nana$",
    #     "ana$",
    #     "na$",
    #     "a$",
    #     "$",
    # ]

    # str_tree = CHSuffixTree(
    #     StringOracle(strings),
    #     len(strings),
    # )

    # str_tree.build()
    # str_tree.validate_leaves()

    # str_tree.dump()

    # print("\n###############################################\n")

    pd = [INF, 1, 2, INF, 1, INF, 1, 2, 1, 2, 3, -1]

    pd_tree = CHSuffixTree(PDOracle(pd), len(pd))

    pd_tree.build()
    pd_tree.validate_leaves()

    pd_tree.dump()
