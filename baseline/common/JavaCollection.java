import java.io.*;
import java.lang.management.*;
import java.nio.file.*;
import java.util.*;

/** Binary collection transport for CT counts. */
public final class JavaCollection {
    static final int INPUT_MAGIC = 0x43544932, OUTPUT_MAGIC = 0x43544f32;

    static long gcMillis() {
        long total = 0;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            if (bean.getCollectionTime() >= 0) total += bean.getCollectionTime();
        }
        return total;
    }

    static final class Row {
        Map<CTMiner.Pattern, double[]> counts;
        double index, count, setup, parse;
        int nodes = -1, candidates = -1;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) throw new IllegalArgumentException("ENGINE CLASSES INPUT OUTPUT");
        String engine = args[0];
        if (!engine.equals("CT"))
            throw new IllegalArgumentException("Unknown collection engine");
        Path classes = Paths.get(args[1]);
        long collectionTick = System.nanoTime(), gcStart = gcMillis();
        double inputSeconds = 0, outputSeconds = 0;
        Map<CTMiner.Pattern, Integer> ids = new HashMap<>();
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(args[2]))));
             DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(args[3]))))) {
            long tick = System.nanoTime();
            if (input.readInt() != INPUT_MAGIC || input.readInt() != 1) throw new IOException("Invalid input version");
            int samples = input.readInt(), nlengths = input.readInt();
            if (samples < 1 || nlengths < 1) throw new IOException("Empty collection or lengths");
            int[] lengths = new int[nlengths];
            for (int i = 0; i < nlengths; i++) lengths[i] = input.readInt();
            lengths = Arrays.stream(lengths).distinct().sorted().toArray();
            if (lengths[0] < 2) throw new IOException("Invalid lengths");
            int bins = input.readInt(); double decay = input.readDouble();
            boolean dropTies = input.readBoolean(), direct = input.readBoolean();
            if (bins < 1 || !Double.isFinite(decay) || decay < 0 || decay > 100)
                throw new IOException("Invalid bins/decay");
            int nwanted = input.readInt();
            Set<CTMiner.Pattern> wanted = nwanted < 0 ? null : new LinkedHashSet<>();
            for (int i = 0; i < nwanted; i++) {
                int m = input.readInt(); if (m < 2) throw new IOException("Invalid pattern length");
                int[] codes = new int[m]; for (int j = 0; j < m; j++) codes[j] = input.readInt();
                wanted.add(CTMiner.Pattern.fromOwnedCodes(codes));
            }
            inputSeconds += (System.nanoTime() - tick) / 1e9;
            output.writeInt(OUTPUT_MAGIC); output.writeInt(1); output.writeInt(samples); output.writeInt(bins);
            for (int sample = 0; sample < samples; sample++) {
                tick = System.nanoTime();
                int n = input.readInt(); if (n < 1) throw new IOException("Empty sequence");
                double[] x = new double[n];
                for (int i = 0; i < n; i++) {
                    x[i] = input.readDouble();
                    if (!Double.isFinite(x[i])) throw new IOException("Nonfinite value");
                }
                inputSeconds += (System.nanoTime() - tick) / 1e9;
                Row row;
                if (engine.equals("CT")) {
                    CTMiner.Result result = CTMiner.mine(x, lengths, bins, decay, dropTies, wanted, direct);
                    row = new Row(); row.counts = result.counts; row.index = result.indexSeconds;
                    row.count = result.countSeconds; row.nodes = result.nodes;
                } else {
                    throw new IllegalArgumentException("Only CT is included");
                }
                tick = System.nanoTime();
                // Define each pattern once per collection. Subsequent rows use IDs.
                for (CTMiner.Pattern p : row.counts.keySet()) if (!ids.containsKey(p)) {
                    int id = ids.size(); ids.put(p, id);
                    output.writeByte(1); output.writeInt(id); output.writeInt(p.length());
                    p.writeCodes(output);
                }
                output.writeByte(2); output.writeInt(sample); output.writeInt(row.counts.size());
                for (Map.Entry<CTMiner.Pattern, double[]> entry : row.counts.entrySet()) {
                    output.writeInt(ids.get(entry.getKey()));
                    for (double value : entry.getValue()) output.writeDouble(value);
                }
                output.writeDouble(row.index); output.writeDouble(row.count);
                output.writeDouble(row.setup); output.writeDouble(row.parse);
                output.writeInt(row.nodes); output.writeInt(row.candidates);
                outputSeconds += (System.nanoTime() - tick) / 1e9;
            }
            if (input.read() != -1) throw new IOException("Trailing input");
            tick = System.nanoTime(); output.flush(); outputSeconds += (System.nanoTime() - tick) / 1e9;
            output.writeByte(3); output.writeDouble(inputSeconds); output.writeDouble(outputSeconds);
            output.writeDouble((System.nanoTime() - collectionTick) / 1e9);
            output.writeLong(gcMillis() - gcStart); output.flush();
        }
    }
}
