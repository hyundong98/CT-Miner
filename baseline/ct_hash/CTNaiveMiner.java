import java.util.*;

/**
 * Independent-window CT baseline. No suffix index, rolling encoding, or counts
 * shared between lengths. Only the immutable key type is shared with CTMiner.
 * Each valid window is encoded from scratch using a monotone stack, then counted.
 */
public final class CTNaiveMiner {
    private CTNaiveMiner() {}

    /**
     * Ordinary overlapping occurrence counts for one sequence (one scalar/key).
     * Duplicate requested lengths are ignored; lengths larger than x are empty.
     * dropTies rejects a window containing ANY repeated value, not just adjacent
     * equal values. Otherwise ties have CTMiner's stable-left minimum semantics.
     * Input and returned immutable pattern keys are never modified.
     *
     * For length m, encoding/hashing work is O((n-m+1)*m), plus map operations;
     * this is the usual expected-time bound under well-distributed hashing.
     * Distinct windows attain that encoding bound even when their CTs coincide.
     * Stored keys use O(D_m*m) space; stack/tie scratch uses O(m).
     */
    public static Map<CTMiner.Pattern, double[]> mine(
            double[] x, int[] lengths, boolean dropTies) {
        return mine(x, lengths, dropTies, Integer.MAX_VALUE);
    }

    /** Same exact counts, with a hard candidate cap (never partial output). */
    public static Map<CTMiner.Pattern, double[]> mine(
            double[] x, int[] lengths, boolean dropTies, int cap) {
        Objects.requireNonNull(x, "sequence");
        Objects.requireNonNull(lengths, "lengths");
        if (cap < 1) throw new IllegalArgumentException("Positive pattern cap required");
        for (double value : x)
            if (!Double.isFinite(value))
                throw new IllegalArgumentException("Nonfinite sequence value");
        int[] requested = Arrays.stream(lengths).distinct().sorted().toArray();
        if (requested.length == 0 || requested[0] < 2)
            throw new IllegalArgumentException("At least one length >= 2 required");
        Map<CTMiner.Pattern, double[]> counts = new HashMap<>();
        for (int m : requested) {
            if (m > x.length) continue;
            int[] stack = new int[m];
            // Scratch storage is reused, but every window is processed afresh.
            Set<Double> seen = dropTies ? new HashSet<>() : null;
            for (int start = 0; start <= x.length - m; start++) {
                if (seen != null) seen.clear();
                int top = 0;
                int[] codes = new int[m];
                boolean valid = true;
                for (int j = 0; j < m; j++) {
                    double value = x[start + j];
                    // Numerical equality treats signed zeros as the same value.
                    if (seen != null && !seen.add(value == 0.0 ? 0.0 : value)) {
                        valid = false;
                        break;
                    }
                    while (top > 0 && x[start + stack[top - 1]] > value) top--;
                    codes[j] = top == 0 ? 0 : j - stack[top - 1];
                    stack[top++] = j;
                }
                if (valid) {
                    // Transfer ownership: no extra clone, and no later mutation.
                    CTMiner.Pattern key = CTMiner.Pattern.fromOwnedCodes(codes);
                    counts.computeIfAbsent(key, ignored -> new double[1])[0]++;
                    if (counts.size() > cap)
                        throw new IllegalStateException("CT naive pattern cap exceeded; no partial output accepted");
                }
            }
        }
        return counts;
    }
}
