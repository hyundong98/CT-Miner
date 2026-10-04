import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.lang.management.*;

/** Same-output Java baselines. Full candidate discovery in ALL three paths.
 * Uses the CTMiner and production top-k comparator/heap.
 * Multiple lengths share each per-sequence Trie build. Queries reuse counts,
 * not a new output-sensitive index. No changes to CTMiner construction.
 */
public final class CTMiningBenchmark {
    static double seconds(long tick) { return (System.nanoTime()-tick)/1e9; }
    static String quote(String s) { return "\""+s.replace("\\", "\\\\").replace("\"", "\\\"")+"\""; }

    static Map<CTMiner.Pattern, double[]> windows(double[] x, int[] lengths, boolean dropTies) {
        return CTNaiveMiner.mine(x, lengths, dropTies);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 6) throw new IllegalArgumentException("INPUT LENGTHS BUDGETS STRATEGY DROP_TIES OUTPUT");
        int[] lengths = Arrays.stream(CTMiner.integers(args[1])).distinct().sorted().toArray();
        int[] budgets = Arrays.stream(CTMiner.integers(args[2])).distinct().sorted().toArray();
        String strategy = args[3]; boolean dropTies = Boolean.parseBoolean(args[4]);
        if (lengths.length == 0 || lengths[0] < 2 || budgets.length == 0 || budgets[0] < 1
                || !Set.of("window_heap", "trie_sort", "trie_heap").contains(strategy))
            throw new IllegalArgumentException("Invalid benchmark settings");
        List<double[]> samples = new ArrayList<>();
        long tick = System.nanoTime(), points = 0;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(args[0]))))) {
            if (in.readInt() != 0x43544233) throw new IOException("Invalid benchmark input");
            int count = in.readInt(); if (count < 1) throw new IOException("Empty input");
            for (int i = 0; i < count; i++) {
                int n = in.readInt(); if (n < 1) throw new IOException("Empty sequence");
                double[] x = new double[n];
                for (int j = 0; j < n; j++) {
                    x[j] = in.readDouble(); if (!Double.isFinite(x[j])) throw new IOException("Nonfinite value");
                }
                samples.add(x); points += n;
            }
            if (in.read() != -1) throw new IOException("Trailing input");
        }
        double inputSeconds = seconds(tick);
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) pool.resetPeakUsage();
        tick = System.nanoTime(); double indexSeconds = 0, countSeconds = 0, aggregateSeconds = 0;
        long nodes = 0;
        Map<CTMiner.Pattern, CTTopKCollection.Entry> aggregate = new HashMap<>();
        for (double[] x : samples) {
            Map<CTMiner.Pattern, double[]> counts;
            if (strategy.equals("window_heap")) {
                long t = System.nanoTime(); counts = windows(x, lengths, dropTies); countSeconds += seconds(t);
            } else {
                CTMiner.Result mined = CTMiner.mine(x, lengths, 1, 0, dropTies, null);
                counts = mined.counts; indexSeconds += mined.indexSeconds; countSeconds += mined.countSeconds; nodes += mined.nodes;
            }
            long t = System.nanoTime();
            for (Map.Entry<CTMiner.Pattern, double[]> entry : counts.entrySet()) {
                CTTopKCollection.Entry total = aggregate.get(entry.getKey());
                if (total == null) {
                    total = new CTTopKCollection.Entry(entry.getKey(), aggregate.size());
                    aggregate.put(entry.getKey(), total);
                }
                total.support += entry.getValue()[0];
            }
            aggregateSeconds += seconds(t);
        }
        double discoverySeconds = seconds(tick);
        tick = System.nanoTime();
        Map<Integer, List<CTTopKCollection.Entry>> byLength = new TreeMap<>();
        for (int m : lengths) byLength.put(m, new ArrayList<>());
        long codeUnits = 0;
        for (CTTopKCollection.Entry e : aggregate.values()) {
            byLength.get(e.pattern.length()).add(e); codeUnits += e.pattern.length();
        }
        double partitionSeconds = seconds(tick), querySeconds = 0;
        List<String> queryJson = new ArrayList<>();
        for (int m : lengths) {
            List<CTTopKCollection.Entry> candidates = byLength.get(m);
            double preparation = 0;
            if (strategy.equals("trie_sort")) {
                long t = System.nanoTime(); candidates.sort(CTTopKCollection.BEST); preparation = seconds(t);
            }
            querySeconds += preparation;
            for (int budget : budgets) {
                long t = System.nanoTime(); List<CTTopKCollection.Entry> selected;
                if (strategy.equals("trie_sort")) {
                    selected = new ArrayList<>(candidates.subList(0, Math.min(budget, candidates.size())));
                } else {
                    PriorityQueue<CTTopKCollection.Entry> heap = new PriorityQueue<>(CTTopKCollection.BEST.reversed());
                    for (CTTopKCollection.Entry e : candidates) CTTopKCollection.offer(heap, e, budget);
                    selected = new ArrayList<>(heap); selected.sort(CTTopKCollection.BEST);
                }
                double selection = seconds(t); querySeconds += selection;
                StringBuilder patterns = new StringBuilder("[");
                for (CTTopKCollection.Entry e : selected) {
                    if (patterns.length() > 1) patterns.append(',');
                    patterns.append("{\"pattern\":").append(e.pattern).append(",\"support\":").append(e.support).append('}');
                }
                patterns.append(']');
                queryJson.add("{\"length\":"+m+",\"budget\":"+budget+",\"candidates\":"+candidates.size()
                    +",\"selected\":"+selected.size()+",\"ranking_prepare_seconds\":"+preparation
                    +",\"selection_seconds\":"+selection+",\"discovery_plus_first_B_seconds\":"
                    +(discoverySeconds+partitionSeconds+preparation+selection)+",\"patterns\":"+patterns+"}");
            }
        }
        List<String> pools = new ArrayList<>();
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP && pool.getPeakUsage() != null)
                pools.add("{\"pool\":"+quote(pool.getName())+",\"peak_used_bytes\":"+pool.getPeakUsage().getUsed()+"}");
        }
        String json = "{\"strategy\":"+quote(strategy)+",\"points\":"+points+",\"samples\":"+samples.size()
            +",\"input_seconds\":"+inputSeconds+",\"index_seconds\":"+indexSeconds+",\"count_seconds\":"+countSeconds
            +",\"aggregate_seconds\":"+aggregateSeconds+",\"discovery_seconds\":"+discoverySeconds
            +",\"partition_seconds\":"+partitionSeconds+",\"all_queries_seconds\":"+querySeconds
            +",\"discovery_and_queries_seconds\":"+(discoverySeconds+partitionSeconds+querySeconds)
            +",\"nodes_built_total\":"+nodes+",\"candidate_patterns\":"+aggregate.size()
            +",\"candidate_code_units\":"+codeUnits+",\"heap_pool_peaks\":["+String.join(",", pools)
            +"],\"queries\":["+String.join(",", queryJson)+"]}";
        Files.writeString(Paths.get(args[5]), json+"\n");
    }
}
