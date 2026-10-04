import java.util.*;

/** Optional archive query layout built from the SAME CTMiner.Trie.
 * Construction algorithm is unchanged; no window-mining fallback is used.
 * Primitive edge intervals replace retained Node/children/suffix-link objects.
 * Counts support strictly increasing length queries, bins=1, decay=0,
 * dropTies=true for both legacy and corrected native CT experiments.
 */
final class CTCompactState {
    final int builtNodes;
    private final int[] pd, validLength, prefix;
    private final int[] source, high, left, right, next, first;
    private final int maxLength;
    private final CTPatternIds ids;
    private final int[] parentEdge, prefixId;
    private int active = -1, previousLength = 0;

    CTCompactState(double[] x, int requestedMaxLength) {
        this(x, requestedMaxLength, null);
    }

    CTCompactState(double[] x, int requestedMaxLength, CTPatternIds ids) {
        this(x, requestedMaxLength, ids, true);
    }

    CTCompactState(double[] x, int requestedMaxLength, CTPatternIds ids, boolean dropTies) {
        if (requestedMaxLength < 2) throw new IllegalArgumentException("Length >= 2 required");
        this.ids = ids;
        int minimumLength = ids == null ? 2 : 1;
        maxLength = Math.min(x.length, requestedMaxLength);
        // tree and all Node references are constructor-local. The builder is
        // retained unchanged so parity isolates the query/storage optimization.
        CTMiner.Trie tree = new CTMiner.Trie(x).build();
        builtNodes = tree.ordered.size();
        pd = tree.pd;
        int[] limits;
        if (dropTies) limits = ids==null ? CTMiner.distinctLimits(x) : CTIntegerTools.distinctLimits(x);
        else { limits = new int[x.length]; for (int i=0;i<x.length;i++) limits[i]=x.length-i; }
        validLength = new int[tree.n];
        prefix = new int[tree.n + 1];
        int cursor = 0;
        for (CTMiner.Node node : tree.ordered) if (node.leaf()) {
            node.left = cursor;
            validLength[cursor++] = node.stringIndex < x.length ? limits[node.stringIndex] : 0;
            node.right = cursor;
        }
        if (cursor != tree.n) throw new IllegalStateException("Invalid CT leaf coverage");
        for (int i = tree.ordered.size() - 1; i >= 0; i--) {
            CTMiner.Node node = tree.ordered.get(i);
            if (!node.leaf()) {
                node.left = tree.n; node.right = 0;
                for (CTMiner.Node child : node.children.values()) {
                    node.left = Math.min(node.left, child.left);
                    node.right = Math.max(node.right, child.right);
                }
            }
        }
        int size = 0;
        for (CTMiner.Node node : tree.ordered)
            if (eligible(node, x.length, maxLength, minimumLength)) size++;
        source = new int[size]; high = new int[size];
        left = new int[size]; right = new int[size]; next = new int[size];
        first = new int[maxLength + 1]; Arrays.fill(first, -1);
        parentEdge = ids == null ? null : new int[size];
        prefixId = ids == null ? null : new int[size];
        // Recover parent edge IDs from DFS preorder with a stack, not hashing.
        CTMiner.Node[] path = ids==null ? null : new CTMiner.Node[size];
        int[] pathIds = ids==null ? null : new int[size];
        int pathSize=0;
        int id = 0;
        for (CTMiner.Node node : tree.ordered) {
            if (!eligible(node, x.length, maxLength, minimumLength)) continue;
            int low = Math.max(minimumLength, node.parent.depth + 1);
            if (ids != null) {
                while (pathSize>0 && path[pathSize-1]!=node.parent) pathSize--;
                if (node.parent!=tree.root && pathSize==0)
                    throw new IllegalStateException("Missing compact CT parent edge");
                parentEdge[id]=pathSize==0 ? -1 : pathIds[pathSize-1];
                path[pathSize]=node; pathIds[pathSize++]=id;
            }
            source[id] = node.source;
            high[id] = Math.min(maxLength, Math.min(node.depth, x.length - node.source));
            left[id] = node.left; right[id] = node.right;
            next[id] = first[low]; first[low] = id++;
        }
    }

    private static boolean eligible(CTMiner.Node node, int n, int maxLength, int minimumLength) {
        return node.parent != null && Math.max(minimumLength, node.parent.depth + 1)
                <= Math.min(maxLength, Math.min(node.depth, n - node.source));
    }

    private void prepare(int length, int cap) {
        if (length <= previousLength || length < (ids==null ? 2 : 1) || length > maxLength || cap < 1
                || (ids!=null && length!=previousLength+1))
            throw new IllegalArgumentException("Increasing lengths within construction bound and positive cap required");
        // Activate an edge exactly once, when the requested length first crosses
        // its parent depth. Gaps in requested lengths are supported.
        for (int depth = previousLength + 1; depth <= length; depth++) {
            for (int id = first[depth]; id != -1;) {
                int following = next[id];
                if (ids != null) {
                    int parent = parentEdge[id];
                    prefixId[id] = parent < 0 ? 0 : prefixId[parent];
                    if (ids.length(prefixId[id])!=depth-1)
                        throw new IllegalStateException("Previous CT depth was not finalized");
                }
                next[id] = active; active = id; id = following;
            }
        }
        previousLength = length;
        // limits[start] <= n-start already enforces the window boundary.
        // One prefix pass replaces separate leaf scans at every queried locus.
        for (int i = 0; i < validLength.length; i++)
            prefix[i + 1] = prefix[i] + (validLength[i] >= length ? 1 : 0);
    }

    Map<CTMiner.Pattern, double[]> counts(int length, int cap) {
        if (ids != null) throw new IllegalStateException("Use level batches for canonical state");
        prepare(length, cap);
        Map<CTMiner.Pattern, double[]> result = new HashMap<>();
        int previous = -1, id = active;
        while (id != -1) {
            int following = next[id];
            if (high[id] < length) {
                if (previous == -1) active = following;
                else next[previous] = following;
            } else {
                int support = prefix[right[id]] - prefix[left[id]];
                if (support > 0) {
                    int[] codes = new int[length];
                    for (int j = 0; j < length; j++) {
                        int code = pd[source[id] + j];
                        codes[j] = code <= j ? code : 0;
                    }
                    CTMiner.Pattern pattern = CTMiner.Pattern.fromOwnedCodes(codes);
                    if (result.put(pattern, new double[]{support}) != null)
                        throw new IllegalStateException("Duplicate CT locus for one pattern");
                    if (result.size() > cap) throw new IllegalStateException("CT pattern cap exceeded");
                }
                previous = id;
            }
            id = following;
        }
        return result;
    }

    /** Emit a whole level. IDs are assigned AFTER every series has emitted. */
    void appendLevel(int length, CTExactCollection.Batch batch, int sample) {
        if (ids == null) throw new IllegalStateException("Canonical registry required");
        prepare(length, Integer.MAX_VALUE);
        int previous = -1, id = active;
        while (id != -1) {
            int following = next[id];
            if (high[id] < length) {
                if (previous == -1) active = following;
                else next[previous] = following;
            } else {
                int support = prefix[right[id]] - prefix[left[id]];
                if (ids.length(prefixId[id])!=length-1)
                    throw new IllegalStateException("Unranked CT prefix");
                int value=pd[source[id]+length-1];
                batch.add(((long)prefixId[id]<<32)|(value<length ? value : 0), sample,id,support);
                previous = id;
            }
            id = following;
        }
    }

    void assign(int edge,int canonicalId) { prefixId[edge]=canonicalId; }
}
