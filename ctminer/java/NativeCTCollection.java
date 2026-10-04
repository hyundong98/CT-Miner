import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Native threshold eligibility with the original STRICT CT-ID representation.
 * One corpus-wide exact ID space and one trie per sample. No per-window
 * re-encoding, per-series output dictionaries, or candidate-length arrays.
 * A pattern qualifies iff at least one sample has support >= minsup. Its
 * ranking score and feature values include counts from ALL samples.
 */
final class NativeCTCollection {
    static void run(double[][] samples,Set<Integer> allowed,int maximum,int budget,int cap,
                    double minsup,long maxCells,Path work,Path output) throws Exception {
        long started=System.nanoTime();
        int nmax=0; for (double[] x:samples) nmax=Math.max(nmax,x.length);
        int bound=Math.min(maximum,nmax);
        CTPatternIds ids=new CTPatternIds(Math.max(1,bound));
        CTCompactState[] states=new CTCompactState[samples.length];
        PriorityQueue<ArchiveJavaCollection.Entry> best=new PriorityQueue<>(ArchiveJavaCollection.BEST.reversed());
        List<ArchiveJavaCollection.Entry> all=new ArrayList<>();
        List<ArchiveJavaCollection.Stage> stages=new ArrayList<>();
        long records=0; int eligibleVisited=0,visited=0;
        double featureSeconds=0.;
        String stop="all_requested_lengths";
        // A single reusable sparse spool replaces one file per sample/length.
        Path spoolPath=Files.createTempFile(work,"ct-native-counts-",".bin");
        try {
            for (int depth=1;depth<=bound;depth++) {
                long levelStart=System.nanoTime();
                ArchiveJavaCollection.Stage stage=new ArchiveJavaCollection.Stage(); stage.length=depth;
                CTExactCollection.Batch batch=new CTExactCollection.Batch();
                for (int s=0;s<samples.length;s++) {
                    if (depth>samples[s].length) { states[s]=null; continue; }
                    long tick=System.nanoTime();
                    if (states[s]==null) {
                        states[s]=new CTCompactState(samples[s],Math.max(2,bound),ids,true);
                        stage.nodes+=states[s].builtNodes;
                        stage.indexSeconds+=(System.nanoTime()-tick)/1e9;
                        tick=System.nanoTime();
                    }
                    states[s].appendLevel(depth,batch,s);
                    stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                }
                records+=batch.size; stage.candidates=batch.size;
                List<ArchiveJavaCollection.Entry> entries;
                long tick=System.nanoTime();
                if (depth==1) {
                    // Depth one establishes exact parent IDs but is not an output.
                    batch.finish(ids,states,null,cap,stage,minsup);
                    stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                    stage.seconds=(System.nanoTime()-levelStart)/1e9; stages.add(stage);
                    continue;
                }
                try (DataOutputStream spool=new DataOutputStream(new BufferedOutputStream(
                        Files.newOutputStream(spoolPath,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)))) {
                    entries=batch.finish(ids,states,spool,cap,stage,minsup);
                }
                stage.countSeconds+=(System.nanoTime()-tick)/1e9-stage.aggregateSeconds;
                visited=depth;
                tick=System.nanoTime();
                int qualifying=0;
                for (ArchiveJavaCollection.Entry e:entries) {
                    stage.observed++; stage.maximum=Math.max(stage.maximum,e.support);
                    if (e.stableSeries==0) continue;
                    qualifying++;
                    if (!allowed.contains(depth)) continue;
                    stage.eligible++;
                    if (eligibleVisited==Integer.MAX_VALUE) throw new IllegalStateException("Too many eligible CT outputs");
                    eligibleVisited++;
                    if (eligibleVisited>cap) throw new IllegalStateException("CT visited eligible output cap exceeded; no truncation");
                    if (budget==0) {
                        if (all.size()>=cap) throw new IllegalStateException("CT eligible output cap exceeded; no truncation");
                        if ((long)(all.size()+1)*samples.length>maxCells)
                            throw new IllegalStateException("Feature matrix exceeds explicit cell limit; not truncated");
                        all.add(e);
                    } else if (best.size()<budget) best.add(e);
                    else if (ArchiveJavaCollection.BEST.compare(e,best.peek())<0) { best.poll(); best.add(e); }
                }
                stage.cutoff=budget>0 && best.size()==budget ? best.peek().support : Double.NaN;
                stage.selectSeconds=(System.nanoTime()-tick)/1e9;
                // Replay the COUNTS already computed by the trie only for new
                // retained columns. Entries retained from prior levels already
                // own their complete values, including below-minsup counts.
                tick=System.nanoTime();
                Collection<ArchiveJavaCollection.Entry> retained=budget==0 ? all : best;
                if ((long)retained.size()*samples.length>maxCells)
                    throw new IllegalStateException("Feature matrix exceeds explicit cell limit; not truncated");
                ArchiveJavaCollection.Entry[] keep=new ArchiveJavaCollection.Entry[entries.size()];
                int newColumns=0;
                // Only this depth can have unallocated columns. Avoid scanning
                // every earlier threshold output at every later depth.
                if (budget==0) {
                    for (ArchiveJavaCollection.Entry e:entries) if (e.stableSeries>0 && allowed.contains(depth)) {
                        e.values=new double[samples.length]; keep[e.id]=e; newColumns++;
                    }
                } else {
                    for (ArchiveJavaCollection.Entry e:best) if (e.values==null) {
                        e.values=new double[samples.length]; keep[e.id]=e; newColumns++;
                    }
                }
                if (newColumns>0) {
                    long bytes=Files.size(spoolPath);
                    if (bytes%16!=0) throw new IOException("Malformed CT sparse counts");
                    try (DataInputStream spool=new DataInputStream(new BufferedInputStream(Files.newInputStream(spoolPath)))) {
                        for (long i=0;i<bytes/16;i++) {
                            int id=spool.readInt(),s=spool.readInt(); double count=spool.readDouble();
                            if (keep[id]!=null) keep[id].values[s]=count;
                        }
                        if (spool.read()!=-1) throw new IOException("Trailing CT sparse counts");
                    }
                }
                stage.replaySeconds=(System.nanoTime()-tick)/1e9; featureSeconds+=stage.replaySeconds;
                stage.seconds=(System.nanoTime()-levelStart)/1e9; stages.add(stage);
                // Every strict CT extension maps to its strict prefix. Per-series
                // and corpus support cannot increase under extension. Neither
                // decision below changes threshold eligibility or final ranking.
                if (qualifying==0) { stop="no_qualifying_prefix"; break; }
                if (budget>0 && best.size()==budget && stage.maximum<best.peek().support*(1.-1e-12)) {
                    stop="unfiltered_support_upper_bound"; break;
                }
            }
        } finally { Files.deleteIfExists(spoolPath); }
        long tick=System.nanoTime();
        List<ArchiveJavaCollection.Entry> selected=budget==0 ? all : new ArrayList<>(best);
        selected.sort(ArchiveJavaCollection.BEST);
        for (ArchiveJavaCollection.Entry e:selected) e.materialize();
        double materializeSeconds=(System.nanoTime()-tick)/1e9;
        double miningSeconds=(System.nanoTime()-started)/1e9-featureSeconds;
        try (DataOutputStream out=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(output)))) {
            out.writeInt(0x434e4f31); out.writeInt(1); out.writeInt(samples.length); out.writeInt(selected.size());
            out.writeDouble(miningSeconds); out.writeDouble(featureSeconds); out.writeLong(records); out.writeLong(0);
            out.writeInt(eligibleVisited); out.writeInt(visited);
            for (ArchiveJavaCollection.Entry e:selected) {
                out.writeInt(e.pattern.length()); e.pattern.writeCodes(out); out.writeDouble(e.support); out.writeInt(e.stableSeries);
                for (double v:e.values) out.writeDouble(v);
            }
        }
        long units=0; for (ArchiveJavaCollection.Entry e:selected) units+=e.pattern.length();
        StringBuilder meta=new StringBuilder("{\"version\":\"ct_native_strict_id_v2\",\"stop_reason\":\"").append(stop)
            .append("\",\"eligible_patterns_visited\":").append(eligibleVisited)
            .append(",\"eligible_union_complete\":").append(!stop.equals("unfiltered_support_upper_bound"))
            .append(",\"materialized_patterns\":").append(selected.size()).append(",\"materialized_code_units\":").append(units)
            .append(",\"prefix_ids\":").append(ids.size()-1).append(",\"steps\":[");
        for (int i=0;i<stages.size();i++) {
            ArchiveJavaCollection.Stage s=stages.get(i); if (i>0) meta.append(',');
            meta.append("{\"length\":").append(s.length).append(",\"observed_patterns\":").append(s.observed)
                .append(",\"eligible_patterns\":").append(s.eligible).append(",\"candidates\":").append(s.candidates)
                .append(",\"index_seconds\":").append(s.indexSeconds).append(",\"count_seconds\":").append(s.countSeconds)
                .append(",\"aggregation_spool_seconds\":").append(s.aggregateSeconds)
                .append(",\"selection_seconds\":").append(s.selectSeconds).append(",\"replay_seconds\":").append(s.replaySeconds)
                .append(",\"seconds\":").append(s.seconds).append('}');
        }
        meta.append("]}\n"); Files.writeString(work.resolve("ct_native_stats.json"),meta.toString());
        ids.report(materializeSeconds,records);
        System.out.println("Native strict CT-ID selected="+selected.size()+" visited="+visited+" stop="+stop);
    }
}
