import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Exact corpus-frequency selection. Mine each sequence once; spool counts,
 * select in Java, and emit only the chosen dictionary and its sparse rows.
 * This does not implement output-sensitive candidate discovery.
 */
public final class CTTopKCollection {
    static final class Entry {
        final CTMiner.Pattern pattern;
        final int id;
        double support;
        Entry(CTMiner.Pattern pattern, int id) { this.pattern = pattern; this.id = id; }
    }

    // Same order as Python (-score, len(pattern), pattern).
    static final Comparator<Entry> BEST = (a, b) -> {
        int c = Double.compare(b.support, a.support);
        if (c == 0) c = Integer.compare(a.pattern.length(), b.pattern.length());
        return c == 0 ? a.pattern.compareCodes(b.pattern) : c;
    };

    static void offer(PriorityQueue<Entry> heap, Entry entry, int budget) {
        if (heap.size() < budget) heap.add(entry);
        else if (BEST.compare(entry, heap.peek()) < 0) {
            heap.poll(); heap.add(entry);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2 && args.length != 4) throw new IllegalArgumentException("INPUT OUTPUT [ENGINE CLASSES]");
        String engine = args.length == 2 ? "CT" : args[2];
        if (!engine.equals("CT"))
            throw new IllegalArgumentException("Only CT is included");
        Path classes = args.length == 4 ? Paths.get(args[3]) : null;
        long started = System.nanoTime(), gcStart = JavaCollection.gcMillis();
        Path outputPath = Paths.get(args[1]);
        Path spoolPath = Files.createTempFile(outputPath.toAbsolutePath().getParent(), "ct-counts-", ".bin");
        try {
            run(Paths.get(args[0]), outputPath, spoolPath, started, gcStart, engine, classes);
        } finally {
            Files.deleteIfExists(spoolPath);
        }
    }

    static void run(Path inputPath, Path outputPath, Path spoolPath,
                    long started, long gcStart, String engine, Path classes) throws Exception {
        Map<CTMiner.Pattern, Entry> entries = new HashMap<>();
        int samples, budget, mode, cap;
        int[] lengths;
        int[] lengthGroups = null, groupBudgets = null;
        double minsup, inputSeconds = 0, aggregateSpoolSeconds = 0;
        long sourceRecords = 0;
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(inputPath)));
             DataOutputStream spool = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(spoolPath)))) {
            long tick = System.nanoTime();
            int magic = input.readInt(), version = input.readInt();
            if (magic != JavaCollection.INPUT_MAGIC || (version != 2 && version != 3))
                throw new IOException("Invalid CT selection input version");
            samples = input.readInt(); int nlengths = input.readInt();
            if (samples < 1 || nlengths < 1) throw new IOException("Empty collection or lengths");
            lengths = new int[nlengths];
            for (int i = 0; i < nlengths; i++) {
                lengths[i] = input.readInt();
                if (lengths[i] < 2 || (i > 0 && lengths[i] <= lengths[i - 1]))
                    throw new IOException("Lengths must be sorted and unique");
            }
            int bins = input.readInt(); double decay = input.readDouble();
            boolean dropTies = input.readBoolean(); input.readBoolean(); // no dictionary lookup in discovery
            if (bins != 1 || !Double.isFinite(decay) || decay < 0 || decay > 100 || input.readInt() != -1)
                throw new IOException("CT selection requires scalar counts and no supplied dictionary");
            budget = input.readInt(); mode = input.readInt(); minsup = input.readDouble(); cap = input.readInt();
            if (budget < 1 || mode < 0 || mode > 3 || (mode == 3) != (version == 3)
                    || !Double.isFinite(minsup) || minsup < 0 || cap < 1)
                throw new IOException("Invalid CT selection options");
            if (mode == 3) {
                int groups = input.readInt();
                if (groups < 1 || groups > nlengths) throw new IOException("Invalid selection groups");
                groupBudgets = new int[groups]; lengthGroups = new int[nlengths]; Arrays.fill(lengthGroups, -1);
                long totalGroupBudget = 0;
                for (int g = 0; g < groups; g++) {
                    groupBudgets[g] = input.readInt(); int count = input.readInt();
                    if (groupBudgets[g] < 1 || count < 1 || count > nlengths) throw new IOException("Invalid group budget/lengths");
                    totalGroupBudget += groupBudgets[g];
                    for (int j = 0; j < count; j++) {
                        int at = Arrays.binarySearch(lengths, input.readInt());
                        if (at < 0 || lengthGroups[at] >= 0) throw new IOException("Overlapping or unknown group length");
                        lengthGroups[at] = g;
                    }
                }
                for (int g : lengthGroups) if (g < 0) throw new IOException("Ungrouped length");
                if (totalGroupBudget > budget) throw new IOException("Group budgets exceed total budget");
            }
            inputSeconds += (System.nanoTime() - tick) / 1e9;
            for (int sample = 0; sample < samples; sample++) {
                tick = System.nanoTime(); int n = input.readInt();
                if (n < 1) throw new IOException("Empty sequence");
                double[] x = new double[n];
                for (int i = 0; i < n; i++) x[i] = input.readDouble(); // CTMiner validates finite values
                inputSeconds += (System.nanoTime() - tick) / 1e9;
                JavaCollection.Row row;
                if (engine.equals("CT")) {
                    CTMiner.Result mined = CTMiner.mine(x, lengths, 1, decay, dropTies, null);
                    row = new JavaCollection.Row(); row.counts = mined.counts;
                    row.index = mined.indexSeconds; row.count = mined.countSeconds; row.nodes = mined.nodes;
                } else {
                    for (double v : x) if (!Double.isFinite(v)) throw new IOException("Nonfinite input");
                    throw new IllegalArgumentException("Only CT is included");
                }
                tick = System.nanoTime();
                spool.writeInt(row.counts.size());
                for (Map.Entry<CTMiner.Pattern, double[]> item : row.counts.entrySet()) {
                    Entry entry = entries.get(item.getKey());
                    if (entry == null) {
                        if (entries.size() >= cap)
                            throw new IOException("Observed vocabulary exceeds max_observed_patterns=" + cap);
                        entry = new Entry(item.getKey(), entries.size()); entries.put(entry.pattern, entry);
                    }
                    double support = item.getValue()[0];
                    // Sample order and scalar additions match Python corpus accumulation.
                    entry.support += support;
                    if (!Double.isFinite(entry.support)) throw new IOException("Nonfinite corpus support");
                    spool.writeInt(entry.id); spool.writeDouble(support);
                }
                spool.writeDouble(row.index); spool.writeDouble(row.count);
                spool.writeDouble(row.setup); spool.writeDouble(row.parse);
                spool.writeInt(row.nodes); spool.writeInt(row.candidates);
                sourceRecords += row.counts.size();
                aggregateSpoolSeconds += (System.nanoTime() - tick) / 1e9;
            }
            if (input.read() != -1) throw new IOException("Trailing input");
            tick = System.nanoTime(); spool.flush();
            aggregateSpoolSeconds += (System.nanoTime() - tick) / 1e9;
        }

        long tick = System.nanoTime();
        int[] observedByLength = new int[lengths.length], eligibleByLength = new int[lengths.length];
        double[] massByLength = new double[lengths.length];
        int eligible = 0;
        List<Entry> selected = new ArrayList<>();
        Map<Integer, PriorityQueue<Entry>> heaps = new HashMap<>();
        for (Entry entry : entries.values()) {
            int at = Arrays.binarySearch(lengths, entry.pattern.length());
            observedByLength[at]++; massByLength[at] += entry.support;
            if (entry.support < minsup) continue;
            eligible++; eligibleByLength[at]++;
            if (mode == 2) selected.add(entry); // explicit all_observed mode
            else {
                int group = mode == 3 ? lengthGroups[at] : (mode == 0 ? 0 : at);
                offer(heaps.computeIfAbsent(group, key -> new PriorityQueue<>(BEST.reversed())),
                        entry, mode == 3 ? groupBudgets[group] : budget);
            }
        }
        for (PriorityQueue<Entry> heap : heaps.values()) selected.addAll(heap);
        selected.sort(BEST);
        int[] selectedIds = new int[entries.size()]; Arrays.fill(selectedIds, -1);
        for (int i = 0; i < selected.size(); i++) selectedIds[selected.get(i).id] = i;
        int observed = entries.size();
        entries.clear(); // retain only selected codes while replaying rows
        double selectionSeconds = (System.nanoTime() - tick) / 1e9;
        long spoolBytes = Files.size(spoolPath);
        double outputReplaySeconds;
        try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(outputPath)));
             DataInputStream spool = new DataInputStream(new BufferedInputStream(Files.newInputStream(spoolPath)))) {
            tick = System.nanoTime();
            output.writeInt(JavaCollection.OUTPUT_MAGIC); output.writeInt(2); output.writeInt(samples); output.writeInt(1);
            for (int i = 0; i < selected.size(); i++) {
                CTMiner.Pattern pattern = selected.get(i).pattern;
                output.writeByte(1); output.writeInt(i); output.writeInt(pattern.length()); pattern.writeCodes(output);
            }
            output.writeByte(4); output.writeInt(selected.size());
            for (Entry entry : selected) output.writeDouble(entry.support);
            output.writeInt(lengths.length);
            for (int i = 0; i < lengths.length; i++) {
                output.writeInt(lengths[i]); output.writeInt(observedByLength[i]);
                output.writeInt(eligibleByLength[i]); output.writeDouble(massByLength[i]);
            }
            output.writeInt(observed); output.writeInt(eligible); output.writeLong(sourceRecords);
            output.writeDouble(aggregateSpoolSeconds); output.writeDouble(selectionSeconds); output.writeLong(spoolBytes);
            int[] rowIds = new int[selected.size()]; double[] rowValues = new double[selected.size()];
            for (int sample = 0; sample < samples; sample++) {
                int records = spool.readInt(), kept = 0;
                for (int j = 0; j < records; j++) {
                    int id = spool.readInt(); double support = spool.readDouble();
                    int selectedId = selectedIds[id];
                    if (selectedId >= 0) {
                        rowIds[kept] = selectedId; rowValues[kept++] = support;
                    }
                }
                output.writeByte(2); output.writeInt(sample); output.writeInt(kept);
                for (int j = 0; j < kept; j++) {
                    output.writeInt(rowIds[j]); output.writeDouble(rowValues[j]);
                }
                for (int j = 0; j < 4; j++) output.writeDouble(spool.readDouble());
                output.writeInt(spool.readInt()); output.writeInt(spool.readInt());
            }
            if (spool.read() != -1) throw new IOException("Trailing spool data");
            output.flush(); outputReplaySeconds = (System.nanoTime() - tick) / 1e9;
            output.writeByte(3); output.writeDouble(inputSeconds); output.writeDouble(outputReplaySeconds);
            output.writeDouble((System.nanoTime() - started) / 1e9);
            output.writeLong(JavaCollection.gcMillis() - gcStart); output.flush();
        }
    }
}
