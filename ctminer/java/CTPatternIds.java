import java.util.*;

/** Exact IDs assigned in lexical order, one complete corpus depth at a time.
 * No hash map, jump table or candidate-length arrays. The caller groups sorted
 * (previous-level ID, next code) records before calling addSorted.
 */
final class CTPatternIds {
    private int[] parent=new int[128], depth=new int[128], code=new int[128];
    private final int maximumLength;
    private int size=1, lastDepth=0; // ID 0 is the empty prefix.
    private long lastKey=-1, materializedPatterns, materializedCodeUnits;

    CTPatternIds(int maximumLength) {
        if (maximumLength < 1) throw new IllegalArgumentException("Positive maximum length required");
        this.maximumLength = maximumLength;
    }

    int length(int id) { return depth[id]; }
    int size() { return size; }

    int addSorted(int previous, int nextCode) {
        if (previous < 0 || previous >= size || depth[previous] >= maximumLength
                || nextCode < 0 || nextCode > depth[previous])
            throw new IllegalArgumentException("Invalid canonical CT extension");
        int d=depth[previous]+1;
        long key=((long)previous<<32)|(nextCode&0xffffffffL);
        if (d<lastDepth || d>lastDepth+1 || (d==lastDepth && key<=lastKey))
            throw new IllegalStateException("Canonical levels must be complete and strictly lexically ordered");
        if (size == depth.length) {
            int capacity = (int) Math.min(Integer.MAX_VALUE - 8L, 2L * size);
            if (capacity <= size) throw new IllegalStateException("Too many CT prefix IDs");
            depth = Arrays.copyOf(depth, capacity); code = Arrays.copyOf(code, capacity);
            parent = Arrays.copyOf(parent, capacity);
        }
        int id = size++;
        depth[id]=d; code[id]=nextCode; parent[id]=previous;
        lastDepth=d; lastKey=key;
        return id;
    }

    /** IDs within a completed depth are exact lexical ranks. */
    int compareCodes(int a, int b) {
        if (a == b) return 0;
        if (depth[a] != depth[b]) throw new IllegalArgumentException("Equal lengths required");
        return Integer.compare(a,b);
    }

    /** Called only for the final output dictionary, after global selection. */
    CTMiner.Pattern materialize(int id) {
        int[] codes = new int[depth[id]];
        materializedPatterns++; materializedCodeUnits += codes.length;
        for (int j = codes.length - 1; j >= 0; j--) {
            codes[j] = code[id]; id = parent[id];
        }
        return CTMiner.Pattern.fromOwnedCodes(codes);
    }

    void report(double materializeSeconds,long rankRecords) {
        System.out.printf(Locale.ROOT,
            "CT_ID_STATS {\"algorithm\":\"level_radix_v2\",\"prefix_ids\":%d,\"rank_records\":%d,\"materialized_patterns\":%d,\"materialized_code_units\":%d,\"final_materialize_seconds\":%.9f}%n",
            size-1,rankRecords,materializedPatterns,materializedCodeUnits,materializeSeconds);
    }
}
