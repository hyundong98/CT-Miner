import java.io.*;
import java.util.*;

/** Every unordered pair of valid corpus windows is tested directly.
 * No pattern hash map, representative-only shortcut, or cached pair decisions.
 * PD preprocessing permits O(length) worst-case exact equivalence comparisons.
 */
final class CTPairwiseMiner {
    private final double[][] samples;
    private final int[][] pd,limits;
    CTPairwiseMiner(double[][] samples) {
        this.samples=samples; pd=new int[samples.length][]; limits=new int[samples.length][];
        for (int s=0;s<samples.length;s++) {
            pd[s]=CTIntegerTools.pd(samples[s]); limits[s]=CTIntegerTools.distinctLimits(samples[s]);
        }
    }
    private int code(int sample,int start,int offset) {
        int value=pd[sample][start+offset]; return value<=offset ? value : 0;
    }
    private boolean equivalent(int a,int startA,int b,int startB,int length) {
        for (int j=0;j<length;j++) if (code(a,startA,j)!=code(b,startB,j)) return false;
        return true;
    }

    List<ArchiveJavaCollection.Entry> counts(int length,int cap,DataOutputStream spool,
            ArchiveJavaCollection.Stage stage) throws IOException {
        long tick=System.nanoTime(),windows=0;
        for (int s=0;s<samples.length;s++)
            for (int i=0;i<=samples[s].length-length;i++) if (limits[s][i]>=length) windows++;
        if (windows>Integer.MAX_VALUE-8L) throw new IllegalStateException("Too many pairwise windows");
        int size=(int)windows;
        int[] sample=new int[size],start=new int[size],leader=new int[size];
        for (int s=0,at=0;s<samples.length;s++) {
            for (int i=0;i<=samples[s].length-length;i++) if (limits[s][i]>=length) {
                sample[at]=s; start[at]=i; leader[at]=at; at++;
            }
        }
        long comparisons=0;
        for (int i=0;i<size;i++) {
            for (int j=0;j<i;j++) {
                comparisons++;
                if (equivalent(sample[i],start[i],sample[j],start[j],length)) leader[i]=leader[j];
            }
        }
        stage.candidates=comparisons; // Pair decisions, not number of patterns.
        stage.countSeconds+=(System.nanoTime()-tick)/1e9;
        tick=System.nanoTime();
        // Aggregate already established equivalence classes by integer sorting.
        long[] keys=new long[size];
        for (int i=0;i<size;i++) keys[i]=((long)leader[i]<<32)|(sample[i]&0xffffffffL);
        int[] order=CTIntegerTools.order(keys,size),samplePatterns=new int[samples.length];
        ArchiveJavaCollection.Entry[] byLeader=new ArchiveJavaCollection.Entry[size];
        List<ArchiveJavaCollection.Entry> entries=new ArrayList<>();
        for (int at=0;at<size;) {
            int end=at+1; long key=keys[order[at]];
            while (end<size && keys[order[end]]==key) end++;
            int representative=(int)(key>>>32),s=(int)key;
            ArchiveJavaCollection.Entry entry=byLeader[representative];
            if (entry==null) {
                if (entries.size()>=cap) throw new IllegalStateException("Corpus pattern cap exceeded; no partial output");
                int[] codes=new int[length];
                for (int j=0;j<length;j++) codes[j]=code(sample[representative],start[representative],j);
                entry=new ArchiveJavaCollection.Entry(CTMiner.Pattern.fromOwnedCodes(codes),entries.size());
                byLeader[representative]=entry; entries.add(entry);
            }
            if (++samplePatterns[s]>cap) throw new IllegalStateException("CT pattern cap exceeded");
            ArchiveJavaCollection.record(spool,entry,s,end-at); at=end;
        }
        stage.aggregateSeconds+=(System.nanoTime()-tick)/1e9;
        System.out.printf(Locale.ROOT,"CT_PAIRWISE length=%d valid_windows=%d comparisons=%d%n",length,size,comparisons);
        return entries;
    }
}
