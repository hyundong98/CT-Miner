import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Binary entry point for strict CT collection mining. A pattern qualifies
 * when at least one series meets minsup; scores and features use all series.
 * Top-K mining uses support bounds for safe early stopping. Threshold mode
 * returns all qualifying patterns, reusing trie counts for feature extraction.
 */
public final class NativePatternCollection {
    static final Set<String> ENGINES=Set.of("CT","CT-ID");
    public static void main(String[] args) throws Exception {
        if (args.length!=4) throw new IllegalArgumentException("ENGINE INPUT OUTPUT WORK");
        String engine=args[0];
        if (!ENGINES.contains(engine)) throw new IllegalArgumentException("Unknown native engine");
        Path work=Paths.get(args[3]); Files.createDirectories(work);
        double[][] samples; Set<Integer> allowed=new HashSet<>();
        int budget,cap,maximum=2; double minsup,minconf,maxCV,decay; long minGaps,maxCells;
        try (DataInputStream in=new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(args[1]))))) {
            if (in.readInt()!=0x434e4931 || in.readInt()!=1) throw new IOException("Native input version mismatch");
            int ns=in.readInt(),nl=in.readInt();
            if (ns<1 || nl<1 || nl>1000000) throw new IOException("Invalid dimensions");
            for (int i=0;i<nl;i++) { int length=in.readInt(); if (length<2 || length>1000000 || !allowed.add(length)) throw new IOException("Invalid length"); maximum=Math.max(maximum,length); }
            budget=in.readInt(); cap=in.readInt(); minsup=in.readDouble(); minconf=in.readDouble(); maxCV=in.readDouble(); decay=in.readDouble(); minGaps=in.readLong(); maxCells=in.readLong();
            if (budget<0 || cap<1 || maxCells<1 || minGaps<1 || !Double.isFinite(minsup) || minsup<=0
                    || !Double.isFinite(minconf) || minconf<0 || minconf>1 || !Double.isFinite(maxCV) || maxCV<0
                    || !Double.isFinite(decay) || decay<0 || decay>100) throw new IOException("Invalid native settings");
            samples=new double[ns][];
            for (int i=0;i<ns;i++) {
                int n=in.readInt(); if (n<1) throw new IOException("Empty sequence"); samples[i]=new double[n];
                for (int j=0;j<n;j++) { double v=in.readDouble(); if (!Double.isFinite(v)) throw new IOException("Nonfinite value"); samples[i][j]=v==0. ? 0. : v; }
            }
            if (in.read()!=-1) throw new IOException("Trailing native input");
        }
        if (engine.equals("CT") || engine.equals("CT-ID")) {
            NativeCTCollection.run(samples,allowed,maximum,budget,cap,minsup,maxCells,work,Paths.get(args[2]));
            return;
        }
    }
}
