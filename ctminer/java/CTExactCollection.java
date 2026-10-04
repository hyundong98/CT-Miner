import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Hash-free post-build CT ranking and the explicit corpus-pairwise baseline.
 * Selection, sparse feature replay and stopping match ArchiveJavaCollection.
 */
final class CTExactCollection {
    static final class Batch {
        long[] key=new long[128];
        int[] sample=new int[128],edge=new int[128],support=new int[128];
        int size;
        void add(long k,int s,int e,int count) {
            if (size==key.length) {
                int capacity=(int)Math.min(Integer.MAX_VALUE-8L,2L*size);
                if (capacity<=size) throw new IllegalStateException("Too many CT level records");
                key=Arrays.copyOf(key,capacity); sample=Arrays.copyOf(sample,capacity);
                edge=Arrays.copyOf(edge,capacity); support=Arrays.copyOf(support,capacity);
            }
            key[size]=k; sample[size]=s; edge[size]=e; support[size++]=count;
        }

        List<ArchiveJavaCollection.Entry> finish(CTPatternIds ids,CTCompactState[] states,
                DataOutputStream spool,int cap,ArchiveJavaCollection.Stage stage) throws IOException {
            return finish(ids,states,spool,cap,stage,0.);
        }

        /** Native qualification is per series; support still includes EVERY series.
         * The legacy overload passes zero and preserves its original metadata.
         */
        List<ArchiveJavaCollection.Entry> finish(CTPatternIds ids,CTCompactState[] states,
                DataOutputStream spool,int cap,ArchiveJavaCollection.Stage stage,double minimumSeriesSupport) throws IOException {
            int[] order=CTIntegerTools.order(key,size),positive=new int[states.length];
            List<ArchiveJavaCollection.Entry> entries=new ArrayList<>();
            for (int at=0;at<size;) {
                int end=at+1; long k=key[order[at]];
                while (end<size && key[order[end]]==k) end++;
                int canonical=ids.addSorted((int)(k>>>32),(int)k);
                boolean nonzero=false; int previousSample=-1;
                for (int j=at;j<end;j++) {
                    int i=order[j],s=sample[i];
                    // Stable grouping preserves sample order within a key.
                    if (s==previousSample) throw new IllegalStateException("Duplicate CT locus in a sample");
                    previousSample=s; states[s].assign(edge[i],canonical);
                    if (support[i]>0) {
                        nonzero=true;
                        if (spool!=null && ++positive[s]>cap) throw new IllegalStateException("CT pattern cap exceeded");
                    }
                }
                if (spool!=null && nonzero) {
                    if (entries.size()>=cap) throw new IllegalStateException("Corpus pattern cap exceeded; no partial output");
                    ArchiveJavaCollection.Entry entry=new ArchiveJavaCollection.Entry(ids,canonical,entries.size());
                    entries.add(entry);
                    long tick=System.nanoTime();
                    for (int j=at;j<end;j++) {
                        int i=order[j];
                        if (support[i]>0) {
                            ArchiveJavaCollection.record(spool,entry,sample[i],support[i]);
                            if (minimumSeriesSupport>0 && support[i]>=minimumSeriesSupport) entry.stableSeries++;
                        }
                    }
                    stage.aggregateSeconds+=(System.nanoTime()-tick)/1e9;
                }
                at=end;
            }
            return entries;
        }
    }

    static ArchiveJavaCollection.Result discover(double[][] samples,String engine,int[] lengths,
            int budget,int cap,Path directory,boolean all,boolean stopEarly) throws Exception {
        return discover(samples,engine,lengths,budget,cap,directory,all,stopEarly,0.0);
    }
    static ArchiveJavaCollection.Result discover(double[][] samples,String engine,int[] lengths,
            int budget,int cap,Path directory,boolean all,boolean stopEarly,double minimumSupport) throws Exception {
        return discover(samples,engine,lengths,budget,cap,directory,all,stopEarly,minimumSupport,true);
    }
    static ArchiveJavaCollection.Result discover(double[][] samples,String engine,int[] lengths,
            int budget,int cap,Path directory,boolean all,boolean stopEarly,double minimumSupport,boolean dropTies) throws Exception {
        long started=System.nanoTime(),gc=JavaCollection.gcMillis();
        boolean ranked=engine.equals("CT-ID");
        CTPatternIds ids=ranked ? new CTPatternIds(lengths[lengths.length-1]) : null;
        CTCompactState[] states=ranked ? new CTCompactState[samples.length] : null;
        CTPairwiseMiner pairwise=null;
        ArchiveJavaCollection.Result result=new ArchiveJavaCollection.Result();
        int processed=0,maxLength=0; long rankRecords=0;
        for (double[] x:samples) maxLength=Math.max(maxLength,x.length);
        for (int length:lengths) {
            if (length>maxLength) { result.stop="sequence_length_bound"; break; }
            long stageStart=System.nanoTime();
            ArchiveJavaCollection.Stage stage=new ArchiveJavaCollection.Stage(); stage.length=length;
            Path spoolPath=Files.createTempFile(directory,"ct-exact-counts-",".bin");
            try {
                List<ArchiveJavaCollection.Entry> entries=new ArrayList<>();
                try (DataOutputStream spool=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(spoolPath)))) {
                    if (ranked) {
                        // Even gapped requests finalize intermediate ranks; no
                        // code-string prefix is reconstructed to cross a gap.
                        for (int depth=processed+1;depth<=length;depth++) {
                            Batch batch=new Batch();
                            for (int s=0;s<samples.length;s++) {
                                if (depth>samples[s].length) { states[s]=null; continue; }
                                long tick=System.nanoTime();
                                if (states[s]==null) {
                                    states[s]=new CTCompactState(samples[s],lengths[lengths.length-1],ids,dropTies);
                                    stage.nodes+=states[s].builtNodes;
                                    stage.indexSeconds+=(System.nanoTime()-tick)/1e9;
                                    tick=System.nanoTime();
                                }
                                states[s].appendLevel(depth,batch,s);
                                stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                            }
                            long tick=System.nanoTime(); double previousAggregation=stage.aggregateSeconds;
                            entries=batch.finish(ids,states,depth==length ? spool : null,cap,stage);
                            stage.countSeconds+=(System.nanoTime()-tick)/1e9-(stage.aggregateSeconds-previousAggregation);
                            rankRecords+=batch.size; stage.candidates+=batch.size;
                        }
                        processed=length;
                    } else {
                        long tick=System.nanoTime();
                        if (pairwise==null) {
                            pairwise=new CTPairwiseMiner(samples);
                            stage.indexSeconds+=(System.nanoTime()-tick)/1e9;
                        }
                        entries=pairwise.counts(length,cap,spool,stage);
                    }
                    long tick=System.nanoTime(); spool.flush(); stage.aggregateSeconds+=(System.nanoTime()-tick)/1e9;
                }
                long tick=System.nanoTime();
                PriorityQueue<ArchiveJavaCollection.Entry> heap=new PriorityQueue<>(ArchiveJavaCollection.BEST.reversed());
                List<ArchiveJavaCollection.Entry> local=new ArrayList<>();
                for (ArchiveJavaCollection.Entry entry:entries) {
                    stage.observed++; stage.maximum=Math.max(stage.maximum,entry.support);
                    if (entry.support<minimumSupport) continue;
                    stage.eligible++;
                    if (all) local.add(entry);
                    else if (heap.size()<budget) heap.add(entry);
                    else if (ArchiveJavaCollection.BEST.compare(entry,heap.peek())<0) { heap.poll(); heap.add(entry); }
                }
                if (!all) local.addAll(heap);
                List<ArchiveJavaCollection.Entry> combined=new ArrayList<>(result.selected);
                combined.addAll(local);
                if (all && combined.size()>cap) throw new IllegalStateException("CT output cap exceeded; no truncation");
                combined.sort(ArchiveJavaCollection.BEST);
                if (!all && combined.size()>budget) combined=new ArrayList<>(combined.subList(0,budget));
                // Stage-local dense IDs allow direct addressing, no keep map.
                ArchiveJavaCollection.Entry[] keep=new ArchiveJavaCollection.Entry[entries.size()];
                for (ArchiveJavaCollection.Entry e:combined) if (e.values==null) {
                    e.values=new double[samples.length]; keep[e.id]=e;
                }
                result.selected=combined;
                stage.cutoff=combined.size()>=budget ? combined.get(budget-1).support : Double.NaN;
                stage.selectSeconds=(System.nanoTime()-tick)/1e9;
                tick=System.nanoTime();
                long bytes=Files.size(spoolPath);
                if (bytes%16!=0) throw new IOException("Malformed CT sparse records");
                try (DataInputStream spool=new DataInputStream(new BufferedInputStream(Files.newInputStream(spoolPath)))) {
                    for (long i=0;i<bytes/16;i++) {
                        int id=spool.readInt(),sample=spool.readInt(); double count=spool.readDouble();
                        if (keep[id]!=null) keep[id].values[sample]=count;
                    }
                    if (spool.read()!=-1) throw new IOException("Trailing CT sparse records");
                }
                stage.replaySeconds=(System.nanoTime()-tick)/1e9;
            } finally { Files.deleteIfExists(spoolPath); }
            stage.seconds=(System.nanoTime()-stageStart)/1e9; result.stages.add(stage);
            if (minimumSupport==0) System.out.printf(Locale.ROOT,"%s length=%d observed=%d eligible=%d retained=%d seconds=%.6f%n",
                engine,length,stage.observed,stage.eligible,result.selected.size(),stage.seconds);
            if (minimumSupport>0 && stage.maximum<minimumSupport) {
                result.stop="minimum_support_bound"; break;
            }
            if (!all && stopEarly && (stage.maximum==0 || (Double.isFinite(stage.cutoff)
                    && stage.maximum<stage.cutoff*(1.-1e-12)))) {
                result.stop="unfiltered_support_upper_bound"; break;
            }
        }
        long tick=System.nanoTime();
        for (ArchiveJavaCollection.Entry entry:result.selected) entry.materialize();
        double materializeSeconds=(System.nanoTime()-tick)/1e9;
        result.seconds=(System.nanoTime()-started)/1e9; result.gcMillis=JavaCollection.gcMillis()-gc;
        if (ids!=null && minimumSupport==0) ids.report(materializeSeconds,rankRecords);
        return result;
    }
}
