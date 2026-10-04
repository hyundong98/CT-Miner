import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Java port of the Python PDOracle and CHSuffixTree reference.
 * Construction preserves scan/rescan, imaginary nodes and back-propagated links.
 * Sparse counts include implicit edge loci, as in the Python reference.
 * No window-by-window CT reconstruction, dense Catalan table or recursive DFS.
 */
public final class CTMiner {
    static final int INF = 999999999;
    enum Kind { REAL, IMAGINARY, BACK_PROPAGATED, LEAF }

    static final class Node {
        Node parent, suffixLink;
        final int source, end, depth;
        int start, leafCount, stringIndex = -1;
        int left, right;
        double mass;
        Kind kind;
        // Preserve Python dict insertion order, including replacing an edge.
        final Map<Integer, Node> children = new LinkedHashMap<>();
        Node(Node parent, int source, int start, int end, int depth, Kind kind) {
            this.parent = parent; this.source = source; this.start = start;
            this.end = end; this.depth = depth; this.kind = kind;
        }
        int edgeLength() { return end - start; }
        boolean leaf() { return kind == Kind.LEAF; }
    }

    static final class Locus {
        final Node node;
        final int offset;
        Locus(Node node, int offset) { this.node = node; this.offset = offset; }
        boolean explicit() { return offset == node.edgeLength(); }
    }

    static final class Rescan {
        final Locus locus;
        final List<Node> encountered;
        Rescan(Locus locus, List<Node> encountered) {
            this.locus = locus; this.encountered = encountered;
        }
    }

    /** Immutable parent-distance key: roots are serialized as zero. */
    public static final class Pattern {
        // Explicit benchmark-only reconstruction of the removed internal copies.
        // Normal runs never set this JVM property.
        private static final boolean BENCHMARK_COPIES = Boolean.getBoolean("ctpm.benchmarkCopies");
        private final int[] codes;
        private final int hash;
        public Pattern(int[] codes) {
            this(codes, true);
        }
        private Pattern(int[] codes, boolean copy) {
            this.codes = copy ? codes.clone() : codes; this.hash = Arrays.hashCode(this.codes);
        }
        // Package-internal ownership transfer: caller must never mutate the array.
        static Pattern fromOwnedCodes(int[] codes) { return new Pattern(codes, BENCHMARK_COPIES); }
        public int[] codes() { return codes.clone(); }
        // Read-only access for Java ports; avoids cloning a key just to inspect ranks.
        int codeAt(int index) { return codes[index]; }
        void writeCodes(DataOutput output) throws IOException {
            int[] values = BENCHMARK_COPIES ? codes() : codes;
            for (int code : values) output.writeInt(code);
        }
        int compareCodes(Pattern other) { return Arrays.compare(codes, other.codes); }
        public int length() { return codes.length; }
        @Override public int hashCode() { return hash; }
        @Override public boolean equals(Object other) {
            return other instanceof Pattern && Arrays.equals(codes, ((Pattern) other).codes);
        }
        @Override public String toString() { return Arrays.toString(codes); }
    }

    static final class Trie {
        final int[] pd;
        final int n;
        final Node root;
        final List<Node> ordered = new ArrayList<>();

        Trie(double[] x) {
            n = x.length + 1;
            pd = new int[n];
            int[] stack = new int[x.length]; int top = 0;
            for (int i = 0; i < x.length; i++) {
                // Strict > gives the same stable-left treatment of ties as Python.
                while (top > 0 && x[stack[top - 1]] > x[i]) top--;
                pd[i] = top == 0 ? INF : i - stack[top - 1];
                stack[top++] = i;
            }
            pd[x.length] = -1;
            root = new Node(null, -1, 0, 0, 0, Kind.REAL);
            root.suffixLink = root;
        }
        int character(int source, int position) {
            int code = pd[source + position];
            return code <= position ? code : INF;
        }
        int length(int source) { return n - source; }

        Node makeExplicit(Locus locus, Kind kind) {
            if (locus.explicit()) return locus.node;
            Node child = locus.node, parent = child.parent;
            int offset = locus.offset;
            if (parent == null || offset <= 0 || offset >= child.edgeLength())
                throw new IllegalStateException("Invalid implicit edge split");
            Node middle = new Node(parent, child.source, child.start,
                    child.start + offset, parent.depth + offset, kind);
            parent.children.put(character(child.source, child.start), middle);
            child.parent = middle; child.start += offset;
            middle.children.put(character(child.source, child.start), child);
            return middle;
        }

        Node nearestLinkedAncestor(Node x) {
            for (Node node = x; node != null; node = node.parent)
                if (node.suffixLink != null) return node;
            return root;
        }

        Locus locusOnPath(Node ancestor, Node descendant, int depth) {
            if (depth < ancestor.depth || depth > descendant.depth)
                throw new IllegalStateException("Depth outside ancestor path");
            if (depth == ancestor.depth) return new Locus(ancestor, ancestor.edgeLength());
            Node child = descendant;
            while (child.parent != null && child.parent.depth >= depth) child = child.parent;
            if (child.parent == null || child.depth < depth)
                throw new IllegalStateException("Missing path child");
            return new Locus(child, depth - child.parent.depth);
        }

        // A pattern can end inside an edge. Its occurrences are exactly the
        // leaves below that edge's child, just as for an explicit locus.
        Node lookup(Pattern pattern) {
            Node node = root; int position = 0;
            while (position < pattern.length()) {
                int first = pattern.codes[position] == 0 ? INF : pattern.codes[position];
                Node child = node.children.get(first);
                if (child == null) return null;
                int take = Math.min(child.edgeLength(), pattern.length() - position);
                for (int j = 0; j < take; j++) {
                    int code = pattern.codes[position + j];
                    if (character(child.source, child.start + j) != (code == 0 ? INF : code)) return null;
                }
                position += take; node = child;
            }
            return node;
        }

        Locus scan(Node node, int source, int position, int remaining) {
            while (remaining > 0) {
                Node child = node.children.get(character(source, position));
                if (child == null) return new Locus(node, node.edgeLength());
                int compareLength = Math.min(remaining, child.edgeLength()), matched = 0;
                while (matched < compareLength && character(source, position + matched)
                        == character(child.source, child.start + matched)) matched++;
                if (matched < compareLength) return new Locus(child, matched);
                if (remaining < child.edgeLength()) return new Locus(child, remaining);
                position += child.edgeLength(); remaining -= child.edgeLength(); node = child;
            }
            return new Locus(node, node.edgeLength());
        }

        Rescan rescan(Node node, int source, int position, int remaining) {
            List<Node> encountered = new ArrayList<>();
            while (remaining > 0) {
                Node child = node.children.get(character(source, position));
                if (child == null) throw new IllegalStateException("Missing rescan edge");
                if (child.edgeLength() > remaining)
                    return new Rescan(new Locus(child, remaining), encountered);
                position += child.edgeLength(); remaining -= child.edgeLength();
                node = child; encountered.add(node);
            }
            return new Rescan(new Locus(node, node.edgeLength()), encountered);
        }

        void backPropagate(Node ancestor, Node x, List<Node> encountered) {
            // Python encountered[1:-1:2], excluding first and last.
            for (int i = 1; i < encountered.size() - 1; i += 2) {
                Node c = encountered.get(i);
                int targetDepth = c.depth + 1;
                if (targetDepth >= x.depth) continue;
                Locus locus = locusOnPath(ancestor, x, targetDepth);
                if (locus.explicit()) {
                    if (locus.node.suffixLink == null) locus.node.suffixLink = c;
                } else {
                    makeExplicit(locus, Kind.BACK_PROPAGATED).suffixLink = c;
                }
            }
        }

        Node establishLink(int source, Node x) {
            if (x == root) return root;
            Node ancestor = nearestLinkedAncestor(x);
            int startPosition = Math.max(0, ancestor.depth - 1);
            int remaining = x.depth - 1 - startPosition;
            if (remaining < 0 || ancestor.suffixLink == null)
                throw new IllegalStateException("Invalid suffix link rescan");
            Rescan scan = rescan(ancestor.suffixLink, source, startPosition, remaining);
            Node link = makeExplicit(scan.locus, Kind.IMAGINARY);
            backPropagate(ancestor, x, scan.encountered);
            x.suffixLink = link;
            return link;
        }

        Node addLeaf(Node parent, int source) {
            int depth = parent.depth, length = length(source);
            if (depth >= length) throw new IllegalStateException("Empty leaf edge");
            int first = character(source, depth);
            if (parent.children.containsKey(first)) throw new IllegalStateException("Duplicate leaf edge");
            Node leaf = new Node(parent, source, depth, length, length, Kind.LEAF);
            leaf.stringIndex = source;
            parent.children.put(first, leaf);
            return leaf;
        }

        Trie build() {
            Node previous = addLeaf(root, 0);
            for (int i = 1; i < n; i++) {
                Node link = establishLink(i, previous.parent);
                int remaining = length(i) - link.depth;
                if (remaining < 0) throw new IllegalStateException("Negative remaining suffix length");
                Node insertion = makeExplicit(scan(link, i, link.depth, remaining), Kind.REAL);
                previous = addLeaf(insertion, i);
                if ((insertion.kind == Kind.IMAGINARY || insertion.kind == Kind.BACK_PROPAGATED)
                        && insertion.children.size() >= 2) insertion.kind = Kind.REAL;
            }
            ArrayDeque<Node> stack = new ArrayDeque<>(); stack.push(root);
            while (!stack.isEmpty()) {
                Node node = stack.pop(); ordered.add(node);
                for (Node child : node.children.values()) stack.push(child);
            }
            for (int i = ordered.size() - 1; i >= 0; i--) {
                Node node = ordered.get(i);
                if (node.leaf()) node.leafCount = 1;
                else for (Node child : node.children.values()) node.leafCount += child.leafCount;
            }
            return this;
        }

        Pattern pattern(Node node, int length) {
            int[] codes = new int[length];
            for (int i = 0; i < length; i++) {
                int code = character(node.source, i);
                codes[i] = code == INF ? 0 : code;
            }
            return Pattern.fromOwnedCodes(codes);
        }
    }

    public static final class Result {
        public final Map<Pattern, double[]> counts;
        public final double indexSeconds, countSeconds;
        public final int nodes;
        Result(Map<Pattern, double[]> counts, double indexSeconds, double countSeconds, int nodes) {
            this.counts = counts; this.indexSeconds = indexSeconds;
            this.countSeconds = countSeconds; this.nodes = nodes;
        }
    }

    static int[] distinctLimits(double[] x) {
        Map<Double, Integer> next = new HashMap<>();
        int end = x.length;
        int[] limits = new int[x.length];
        for (int i = x.length - 1; i >= 0; i--) {
            // Python regards -0.0 and +0.0 as the same dictionary key.
            double key = x[i] == 0.0 ? 0.0 : x[i];
            end = Math.min(end, next.getOrDefault(key, x.length));
            limits[i] = end - i; next.put(key, i);
        }
        return limits;
    }

    public static Result mine(double[] x, int[] lengths, int bins, double decay,
                              boolean dropTiedWindows, Set<Pattern> wanted) {
        return mine(x, lengths, bins, decay, dropTiedWindows, wanted, true);
    }

    public static Result mine(double[] x, int[] lengths, int bins, double decay,
                              boolean dropTiedWindows, Set<Pattern> wanted, boolean directLookup) {
        if (x.length == 0 || x.length >= INF || bins < 1 || lengths.length == 0
                || !Double.isFinite(decay) || decay < 0 || decay > 100)
            throw new IllegalArgumentException("Nonempty input, positive bins/lengths, decay in [0,100] required");
        for (double value : x) if (!Double.isFinite(value)) throw new IllegalArgumentException("Nonfinite input");
        int[] requested = Arrays.stream(lengths).distinct().sorted().toArray();
        if (requested[0] < 2) throw new IllegalArgumentException("Pattern length must be >= 2");
        long tick = System.nanoTime();
        Trie tree = new Trie(x).build();
        long indexed = System.nanoTime();
        boolean positions = bins > 1 || dropTiedWindows;
        int[] starts = positions ? new int[tree.n] : null;
        if (positions) {
            int cursor = 0;
            // ordered is DFS preorder: subtree leaves form a contiguous interval.
            for (Node node : tree.ordered) if (node.leaf()) {
                node.left = cursor; starts[cursor++] = node.stringIndex; node.right = cursor;
            }
            for (int i = tree.ordered.size() - 1; i >= 0; i--) {
                Node node = tree.ordered.get(i);
                if (!node.leaf()) {
                    node.left = tree.n; node.right = 0;
                    for (Node child : node.children.values()) {
                        node.left = Math.min(node.left, child.left);
                        node.right = Math.max(node.right, child.right);
                    }
                }
            }
        } else if (decay != 0) {
            for (int i = tree.ordered.size() - 1; i >= 0; i--) {
                Node node = tree.ordered.get(i);
                if (node.leaf()) node.mass = node.stringIndex < x.length
                        ? Math.exp(-decay * (x.length - node.stringIndex - 1) / x.length) : 0;
                else for (Node child : node.children.values()) node.mass += child.mass;
            }
        }
        int[] limits = dropTiedWindows ? distinctLimits(x) : null;
        Map<Pattern, double[]> counts = new LinkedHashMap<>();
        if (wanted != null && directLookup) {
            for (Pattern pattern : wanted) {
                int length = pattern.length();
                if (length > x.length || Arrays.binarySearch(requested, length) < 0) continue;
                Node node = tree.lookup(pattern);
                if (node != null) addCount(counts, pattern, node, x.length, bins, decay, starts, limits);
            }
        } else {
            for (Node node : tree.ordered) {
                if (node.parent == null) continue;
                int lo = node.parent.depth + 1, hi = Math.min(node.depth, x.length - node.source);
                for (int length : requested) {
                    if (length < lo) continue;
                    if (length > hi) break;
                    Pattern pattern = tree.pattern(node, length);
                    if (wanted != null && !wanted.contains(pattern)) continue;
                    addCount(counts, pattern, node, x.length, bins, decay, starts, limits);
                }
            }
        }
        return new Result(counts, (indexed - tick) / 1e9,
                (System.nanoTime() - indexed) / 1e9, tree.ordered.size());
    }

    static void addCount(Map<Pattern, double[]> counts, Pattern pattern, Node node,
                         int n, int bins, double decay, int[] starts, int[] limits) {
        int length = pattern.length();
        double[] values = new double[bins];
        if (starts != null) {
            int nwin = n - length + 1;
            for (int j = node.left; j < node.right; j++) {
                int start = starts[j];
                if (start >= nwin || (limits != null && limits[start] < length)) continue;
                int bin = (int) Math.min(bins - 1L, (long) start * bins / nwin);
                values[bin] += decay == 0 ? 1.0 : Math.exp(-decay * (n - start - length) / n);
            }
        } else {
            values[0] = decay == 0 ? node.leafCount : node.mass * Math.exp(decay * (length - 1) / n);
        }
        boolean positive = false;
        for (double value : values) if (value > 0) { positive = true; break; }
        if (positive && counts.put(pattern, values) != null)
            throw new IllegalStateException("Duplicate CT locus for one pattern");
    }

    static double[] readInput(Path path) throws IOException {
        double[] data = new double[1024]; int size = 0;
        try (BufferedReader input = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = input.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                for (String token : line.trim().split("\\s+")) {
                    if (size == data.length) data = Arrays.copyOf(data, Math.multiplyExact(size, 2));
                    data[size++] = Double.parseDouble(token);
                }
            }
        }
        return Arrays.copyOf(data, size);
    }

    static int[] integers(String text) {
        return Arrays.stream(text.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    }

    /** Used by the same ReproBatch JVM/classloader harness as the OP miners. */
    public static void main(String[] args) throws Exception {
        if (args.length != 6 && args.length != 7)
            throw new IllegalArgumentException("INPUT LENGTHS_CSV BINS DECAY TIES WANTED_FILE_OR_DASH [DIRECT_LOOKUP]");
        if (!args[4].equals("native") && !args[4].equals("drop_windows"))
            throw new IllegalArgumentException("Unknown tie policy");
        Set<Pattern> wanted = null;
        if (!args[5].equals("-")) {
            wanted = new HashSet<>();
            for (String line : Files.readAllLines(Paths.get(args[5]), StandardCharsets.UTF_8))
                if (!line.trim().isEmpty()) wanted.add(Pattern.fromOwnedCodes(integers(line)));
        }
        Result result = mine(readInput(Paths.get(args[0])), integers(args[1]),
                Integer.parseInt(args[2]), Double.parseDouble(args[3]),
                args[4].equals("drop_windows"), wanted, args.length == 6 || Boolean.parseBoolean(args[6]));
        // Separate compute time from serialization/JVM startup, just as OP adapters do.
        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8));
        output.write("C\t" + result.indexSeconds + "\t" + result.countSeconds + "\t" + result.nodes + "\n");
        for (Map.Entry<Pattern, double[]> entry : result.counts.entrySet()) {
            output.write("V\t" + entry.getKey() + "\t");
            double[] values = entry.getValue();
            for (int i = 0; i < values.length; i++) {
                if (i > 0) output.write(",");
                output.write(Double.toString(values[i]));
            }
            output.write("\n");
        }
        output.flush(); // ReproBatch owns and closes System.out for this job.
    }
}
