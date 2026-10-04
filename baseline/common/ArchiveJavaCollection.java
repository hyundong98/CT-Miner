import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Shared CT records, sparse feature replay and internal CT diagnostics. */
public final class ArchiveJavaCollection {
    static final int INPUT=0x43414a31, OUTPUT=0x43414f31;
    static final Set<String> ENGINES=Set.of("CT","CT-Compact","CT-ID","CT-Pairwise","CT-Hash");
    static final class Entry {
        CTMiner.Pattern pattern; final int id;
        CTPatternIds canonical;
        final int canonicalId;
        double support, gap, gap2; long ngap;
        int stableSeries;
        double[] values;
        Entry(CTMiner.Pattern p,int id) { pattern=p; this.id=id; canonicalId=-1; }
        Entry(CTPatternIds canonical,int canonicalId,int id) {
            this.canonical=canonical; this.canonicalId=canonicalId; this.id=id;
        }
        int length() { return canonical==null ? pattern.length() : canonical.length(canonicalId); }
        int compareCodes(Entry other) {
            if (canonical==null && other.canonical==null) return pattern.compareCodes(other.pattern);
            if (canonical!=null && canonical==other.canonical)
                return canonical.compareCodes(canonicalId,other.canonicalId);
            throw new IllegalStateException("Mixed CT key spaces in one ranking");
        }
        void materialize() {
            if (canonical!=null) {
                pattern=canonical.materialize(canonicalId);
                canonical=null; // Final results do not retain the prefix registry.
            }
        }
        double cv() { return ngap>0 && gap>0 ? Math.sqrt(Math.max(0.,gap2/ngap-Math.pow(gap/ngap,2)))/(gap/ngap) : Double.NaN; }
    }
    static final Comparator<Entry> BEST=(a,b)-> {
        int c=Double.compare(b.support,a.support);
        if (c==0) c=Integer.compare(a.length(),b.length());
        return c==0 ? a.compareCodes(b) : c;
    };
    static final Comparator<Entry> COVERAGE_BEST=(a,b)-> {
        int c=Integer.compare(b.stableSeries,a.stableSeries);
        return c==0 ? BEST.compare(a,b) : c;
    };
    static final class Stage {
        int length, observed, eligible;
        long nodes, candidates;
        double maximum, cutoff, indexSeconds, countSeconds, setupSeconds, aggregateSeconds, selectSeconds, replaySeconds, seconds;
    }
    static final class Result {
        List<Entry> selected=new ArrayList<>();
        List<Stage> stages=new ArrayList<>();
        double seconds; long gcMillis; String stop="all_requested_lengths";
    }
    /** Cache precisely the existing CH trie and its occurrence intervals. */
    static final class CTState {
        final CTMiner.Trie tree;
        final int[] starts, limits;
        final int n;
        CTState(double[] x) {
            n=x.length; tree=new CTMiner.Trie(x).build();
            starts=new int[tree.n]; limits=CTMiner.distinctLimits(x);
            int cursor=0;
            for (CTMiner.Node node:tree.ordered) if (node.leaf()) {
                node.left=cursor; starts[cursor++]=node.stringIndex; node.right=cursor;
            }
            for (int i=tree.ordered.size()-1; i>=0; i--) {
                CTMiner.Node node=tree.ordered.get(i);
                if (!node.leaf()) {
                    node.left=tree.n; node.right=0;
                    for (CTMiner.Node child:node.children.values()) {
                        node.left=Math.min(node.left,child.left); node.right=Math.max(node.right,child.right);
                    }
                }
            }
        }
        Map<CTMiner.Pattern,double[]> counts(int length,int cap) {
            Map<CTMiner.Pattern,double[]> counts=new HashMap<>();
            for (CTMiner.Node node:tree.ordered) {
                if (node.parent==null || length<=node.parent.depth || length>node.depth || length>n-node.source) continue;
                CTMiner.addCount(counts,tree.pattern(node,length),node,n,1,0.,starts,limits);
                if (counts.size()>cap) throw new IllegalStateException("CT pattern cap exceeded");
            }
            return counts;
        }
    }
    static Entry aggregate(Map<CTMiner.Pattern,Entry> entries,CTMiner.Pattern pattern,int cap) {
        Entry e=entries.get(pattern);
        if (e==null) {
            if (entries.size()>=cap) throw new IllegalStateException("Corpus pattern cap exceeded; no partial top-k accepted");
            e=new Entry(pattern,entries.size()); entries.put(pattern,e);
        }
        return e;
    }
    static void record(DataOutputStream spool,Entry entry,int sample,double count) throws IOException {
        if (!(count>0) || !Double.isFinite(count)) throw new IllegalStateException("Invalid support");
        entry.support+=count;
        if (!Double.isFinite(entry.support)) throw new IllegalStateException("Support overflow");
        spool.writeInt(entry.id); spool.writeInt(sample); spool.writeDouble(count);
    }
    static Result discover(double[][] samples,String engine,Path classes,int[] lengths,int budget,int cap,
                           double maxCV,long minGaps,Path directory,boolean all,boolean stopEarly,
                           boolean coverage,long minSupport,boolean seriesSupport) throws Exception {
        if (engine.equals("CT-ID") || engine.equals("CT-Pairwise"))
            return CTExactCollection.discover(samples,engine,lengths,budget,cap,directory,all,stopEarly);
        long started=System.nanoTime(), gc=JavaCollection.gcMillis();
        Object[] states=new Object[samples.length];
        Result result=new Result();
        Comparator<Entry> ranking=coverage ? COVERAGE_BEST : BEST;
        int maxLength=0; for (double[] x:samples) maxLength=Math.max(maxLength,x.length);
        for (int length:lengths) {
            if (length>maxLength) { result.stop="sequence_length_bound"; break; }
            long stageStart=System.nanoTime(); Stage stage=new Stage(); stage.length=length;
            Map<CTMiner.Pattern,Entry> entries=new HashMap<>();
            Path spoolPath=Files.createTempFile(directory,"archive-counts-",".bin");
            long records=0;
            try {
                try (DataOutputStream spool=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(spoolPath)))) {
                    for (int sample=0; sample<samples.length; sample++) {
                        double[] x=samples[sample]; if (length>x.length) { states[sample]=null; continue; }
                        long tick=System.nanoTime();
                        Map<CTMiner.Pattern,double[]> counts=null;
                        NavigableMap<CTMiner.Pattern,int[]> occurrences=null;
                        if (engine.equals("CT-Hash")) {
                            counts=CTNaiveMiner.mine(x,new int[]{length},true,cap);
                            stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                        } else if (engine.equals("CT-Compact")) {
                            if (states[sample]==null) {
                                states[sample]=new CTCompactState(x,lengths[lengths.length-1]);
                                stage.indexSeconds+=(System.nanoTime()-tick)/1e9;
                                stage.nodes+=((CTCompactState)states[sample]).builtNodes;
                                tick=System.nanoTime();
                            }
                            counts=((CTCompactState)states[sample]).counts(length,cap);
                            stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                        } else if (engine.equals("CT")) {
                            if (states[sample]==null) {
                                states[sample]=new CTState(x);
                                stage.indexSeconds+=(System.nanoTime()-tick)/1e9;
                                stage.nodes+=((CTState)states[sample]).tree.ordered.size(); tick=System.nanoTime();
                            }
                            counts=((CTState)states[sample]).counts(length,cap);
                            stage.countSeconds+=(System.nanoTime()-tick)/1e9;
                        } else { throw new IllegalArgumentException("Unknown CT engine"); }
                        tick=System.nanoTime();
                        if (occurrences!=null) {
                            for (Map.Entry<CTMiner.Pattern,int[]> item:occurrences.entrySet()) {
                                Entry entry=aggregate(entries,item.getKey(),cap); int[] ends=item.getValue();
                                record(spool,entry,sample,ends.length); records++;
                                entry.ngap+=ends.length-1;
                                double localGap=0.,localGap2=0.;
                                for (int j=1; j<ends.length; j++) {
                                    double gap=ends[j]-ends[j-1]; entry.gap+=gap; entry.gap2+=gap*gap;
                                    localGap+=gap; localGap2+=gap*gap;
                                }
                                long n=ends.length-1;
                                if (n>=minGaps && localGap>0. && (!seriesSupport || ends.length>=minSupport)) {
                                    double mean=localGap/n;
                                    double cv=Math.sqrt(Math.max(0.,localGap2/n-mean*mean))/mean;
                                    if (cv<=maxCV) entry.stableSeries++;
                                }
                            }
                        } else {
                            for (Map.Entry<CTMiner.Pattern,double[]> item:counts.entrySet()) {
                                record(spool,aggregate(entries,item.getKey(),cap),sample,item.getValue()[0]); records++;
                            }
                        }
                        stage.aggregateSeconds+=(System.nanoTime()-tick)/1e9;
                    }
                    long tick=System.nanoTime(); spool.flush(); stage.aggregateSeconds+=(System.nanoTime()-tick)/1e9;
                }
                long tick=System.nanoTime();
                PriorityQueue<Entry> heap=new PriorityQueue<>(ranking.reversed());
                List<Entry> local=new ArrayList<>();
                for (Entry e:entries.values()) {
                    stage.observed++; stage.maximum=Math.max(stage.maximum,e.support);
                    stage.eligible++;
                    if (all) local.add(e);
                    else if (heap.size()<budget) heap.add(e);
                    else if (ranking.compare(e,heap.peek())<0) { heap.poll(); heap.add(e); }
                }
                if (!all) local.addAll(heap);
                List<Entry> combined=new ArrayList<>(result.selected); combined.addAll(local); combined.sort(ranking);
                if (!all && combined.size()>budget) combined=new ArrayList<>(combined.subList(0,budget));
                // New entries have no feature vector yet. Replay only globally retained columns.
                Map<Integer,Entry> keep=new HashMap<>();
                for (Entry e:combined) if (e.values==null) { e.values=new double[samples.length]; keep.put(e.id,e); }
                result.selected=combined;
                stage.cutoff=combined.size()>=budget ? (coverage ? combined.get(budget-1).stableSeries : combined.get(budget-1).support) : Double.NaN;
                entries.clear(); local.clear(); heap.clear();
                stage.selectSeconds=(System.nanoTime()-tick)/1e9;
                tick=System.nanoTime();
                try (DataInputStream spool=new DataInputStream(new BufferedInputStream(Files.newInputStream(spoolPath)))) {
                    for (long i=0; i<records; i++) {
                        int id=spool.readInt(),sample=spool.readInt(); double count=spool.readDouble();
                        Entry e=keep.get(id); if (e!=null) e.values[sample]=count;
                    }
                    if (spool.read()!=-1) throw new IOException("Trailing spool data");
                }
                stage.replaySeconds=(System.nanoTime()-tick)/1e9;
            } finally { Files.deleteIfExists(spoolPath); }
            stage.seconds=(System.nanoTime()-stageStart)/1e9; result.stages.add(stage);
            System.out.printf(Locale.ROOT,"%s length=%d observed=%d eligible=%d retained=%d seconds=%.6f%n",
                engine,length,stage.observed,stage.eligible,result.selected.size(),stage.seconds);
            // Stable descendants can have unstable parents; bound by ALL OP support.
            // Stable-series coverage is NOT anti-monotone. A stable child can
            // have an unstable parent. A support cutoff cannot bound coverage.
            if (!all && stopEarly && (stage.maximum==0 || (!coverage && Double.isFinite(stage.cutoff) && stage.maximum<stage.cutoff*(1.-1e-12)))) {
                result.stop="unfiltered_support_upper_bound"; break;
            }
        }
        // Restore ONLY the final dictionary; include restoration in mining time.
        // For all_patterns=true every returned pattern necessarily needs output.
        for (Entry entry:result.selected) entry.materialize();
        result.seconds=(System.nanoTime()-started)/1e9; result.gcMillis=JavaCollection.gcMillis()-gc;
        return result;
    }
    static void parity(Result expected,Result actual) {
        if (expected.selected.size()!=actual.selected.size()) throw new IllegalStateException("Repeat dictionary size mismatch");
        for (int i=0; i<expected.selected.size(); i++) {
            Entry a=expected.selected.get(i),b=actual.selected.get(i);
            if (!a.pattern.equals(b.pattern) || a.support!=b.support || a.stableSeries!=b.stableSeries || a.ngap!=b.ngap || a.gap!=b.gap || a.gap2!=b.gap2 || !Arrays.equals(a.values,b.values))
                throw new IllegalStateException("Repeat dictionary/features mismatch");
        }
    }
    public static void main(String[] args) throws Exception {
        if (args.length!=8) throw new IllegalArgumentException("ENGINE CLASSES INPUT OUTPUT WARMUPS REPEATS ALL STOP_EARLY");
        String engine=args[0]; if (!ENGINES.contains(engine)) throw new IllegalArgumentException("Unknown miner");
        int warmups=Integer.parseInt(args[4]),repeats=Integer.parseInt(args[5]);
        boolean all=Boolean.parseBoolean(args[6]),stopEarly=Boolean.parseBoolean(args[7]);
        if (warmups<0 || repeats<1) throw new IllegalArgumentException("Invalid repeat counts");
        Path output=Paths.get(args[3]),directory=output.toAbsolutePath().getParent();
        long inputStart=System.nanoTime();
        double[][] samples; int[] lengths; int budget,cap; double maxCV; long minGaps,minSupport;
        boolean coverage,seriesSupport;
        try (DataInputStream in=new DataInputStream(new BufferedInputStream(Files.newInputStream(Paths.get(args[2]))))) {
            if (in.readInt()!=INPUT) throw new IOException("Archive Java input magic; rebuild required");
            int version=in.readInt();
            if (version!=2 && version!=3) throw new IOException("Archive Java input version; rebuild required");
            if (version==3 && !engine.equals("CT-ID") && !engine.equals("CT-Hash"))
                throw new IOException("Long-length input supports CT-ID/CT-Hash only");
            int maximumLength=version==3 ? 1000000 : 63;
            int ns=in.readInt(),nl=in.readInt();
            if (ns<1 || nl<1 || nl>maximumLength-1) throw new IOException("Invalid collection dimensions");
            lengths=new int[nl];
            for (int i=0; i<nl; i++) {
                lengths[i]=in.readInt();
                if (lengths[i]<2 || lengths[i]>maximumLength || (i>0 && lengths[i]<=lengths[i-1])) throw new IOException("Invalid lengths");
            }
            budget=in.readInt(); cap=in.readInt(); maxCV=in.readDouble(); minGaps=in.readLong();
            if (budget<1 || cap<1 || !Double.isFinite(maxCV) || maxCV<0 || minGaps<1) throw new IOException("Invalid selection settings");
            int selection=in.readInt(),scope=in.readInt(); minSupport=in.readLong();
            if (selection<0 || selection>1 || scope<0 || scope>1 || minSupport<1) throw new IOException("Invalid reserved policy fields");
            coverage=false; seriesSupport=scope==1;
            samples=new double[ns][];
            for (int i=0; i<ns; i++) {
                int n=in.readInt(); if (n<1 || n>=CTMiner.INF) throw new IOException("Invalid sequence length"); samples[i]=new double[n];
                for (int j=0; j<n; j++) { samples[i][j]=in.readDouble(); if (!Double.isFinite(samples[i][j])) throw new IOException("Nonfinite value"); }
            }
            if (in.read()!=-1) throw new IOException("Trailing archive input");
        }
        double inputSeconds=(System.nanoTime()-inputStart)/1e9;
        Result reference=null;
        for (int i=0; i<warmups; i++) {
            Result current=discover(samples,engine,Paths.get(args[1]),lengths,budget,cap,maxCV,minGaps,directory,all,stopEarly,coverage,minSupport,seriesSupport);
            if (reference!=null) parity(reference,current); reference=current;
        }
        List<Result> measured=new ArrayList<>();
        for (int i=0; i<repeats; i++) {
            Result current=discover(samples,engine,Paths.get(args[1]),lengths,budget,cap,maxCV,minGaps,directory,all,stopEarly,coverage,minSupport,seriesSupport);
            if (reference!=null) parity(reference,current);
            reference=current; measured.add(current);
        }
        Result selected=measured.get(0);
        try (DataOutputStream out=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(output)))) {
            out.writeInt(OUTPUT); out.writeInt(2); out.writeInt(samples.length); out.writeInt(selected.selected.size());
            out.writeDouble(inputSeconds); out.writeInt(warmups); out.writeInt(repeats);
            for (Result r:measured) {
                out.writeDouble(r.seconds); out.writeLong(r.gcMillis); out.writeUTF(r.stop); out.writeInt(r.stages.size());
                for (Stage s:r.stages) {
                    out.writeInt(s.length); out.writeInt(s.observed); out.writeInt(s.eligible);
                    out.writeLong(s.nodes); out.writeLong(s.candidates);
                    for (double v:new double[]{s.maximum,s.cutoff,s.indexSeconds,s.countSeconds,s.setupSeconds,s.aggregateSeconds,s.selectSeconds,s.replaySeconds,s.seconds}) out.writeDouble(v);
                }
            }
            for (Entry e:selected.selected) {
                out.writeInt(e.pattern.length()); e.pattern.writeCodes(out);
                out.writeDouble(e.support); out.writeLong(e.ngap); out.writeDouble(e.gap); out.writeDouble(e.gap2);
                out.writeInt(e.stableSeries);
                for (double value:e.values) out.writeDouble(value);
            }
        }
    }
}
