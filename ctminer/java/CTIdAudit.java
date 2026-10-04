import java.nio.file.*;
import java.util.*;

/** Exactness checks for the integer CT representation. */
public final class CTIdAudit {
    static void check(double[][] samples,int[] lengths,int budget,boolean all,Path work) throws Exception {
        ArchiveJavaCollection.Result expected=null;
        for (String engine:new String[]{"CT-Hash","CT","CT-Compact","CT-ID","CT-Pairwise"}) {
            ArchiveJavaCollection.Result actual=ArchiveJavaCollection.discover(samples,engine,Paths.get("."),
                lengths,budget,200000,.5,1,work,all,!all,false,2,true);
            if (expected==null) { expected=actual; continue; }
            ArchiveJavaCollection.parity(expected,actual);
            if (!expected.stop.equals(actual.stop) || expected.stages.size()!=actual.stages.size())
                throw new AssertionError("Different stopping decisions: "+engine);
            for (int i=0;i<expected.stages.size();i++) {
                ArchiveJavaCollection.Stage a=expected.stages.get(i),b=actual.stages.get(i);
                if (a.length!=b.length || a.observed!=b.observed || a.eligible!=b.eligible
                        || a.maximum!=b.maximum || Double.compare(a.cutoff,b.cutoff)!=0)
                    throw new AssertionError("Different stage counts/ranking: "+engine+" length="+a.length);
                if (engine.equals("CT-Pairwise")) {
                    long windows=0;
                    for (double[] x:samples) {
                        // Independent tie check for the audit's bounded inputs.
                        for (int start=0;start<=x.length-b.length;start++) {
                            boolean valid=true;
                            for (int u=0;u<b.length;u++) for (int v=0;v<u;v++)
                                if (x[start+u]==x[start+v]) valid=false;
                            if (valid) windows++;
                        }
                    }
                    if (b.candidates!=windows*(windows-1)/2)
                        throw new AssertionError("Pairwise baseline skipped pairs");
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length!=0) throw new IllegalArgumentException("No arguments");
        List<double[]> samples=new ArrayList<>();
        samples.add(new double[]{-0.,0.,1.,-1.,-0.,2.});
        samples.add(new double[]{1.,2.,3.,4.,5.,6.,7.,8.,9.,10.});
        samples.add(new double[]{10.,9.,8.,7.,6.,5.,4.,3.,2.,1.});
        // Exhaustive short ternary inputs include non-adjacent ties/zero support.
        for (int n=2,power=9;n<=5;n++,power*=3) {
            for (int value=0;value<power;value++) {
                double[] x=new double[n]; int digits=value;
                for (int i=0;i<n;i++) { x[i]=digits%3-1; digits/=3; }
                samples.add(x);
            }
        }
        Random random=new Random(17);
        for (int r=0;r<64;r++) {
            double[] x=new double[10+random.nextInt(15)];
            for (int i=0;i<x.length;i++) x[i]=r%2==0 ? random.nextDouble() : random.nextInt(9);
            samples.add(x);
        }
        double[][] corpus=samples.toArray(new double[0][]);
        double[][] reversed=corpus.clone();
        Collections.reverse(Arrays.asList(reversed));
        Path work=Files.createTempDirectory("ct-id-audit-");
        try {
            // Check adjacent, skipped and cold long lengths plus complete output.
            for (int[] lengths:new int[][]{{2,3,4,5,6,7,8},{3,6,9},{8}})
                for (int budget:new int[]{1,5,30}) check(corpus,lengths,budget,false,work);
            check(corpus,new int[]{2,4,7},30,true,work);
            check(reversed,new int[]{2,3,4,5,6,7,8},30,false,work);
            // Lexical ranks and output must not depend on sample encounter order.
            ArchiveJavaCollection.Result a=ArchiveJavaCollection.discover(corpus,"CT-ID",Paths.get("."),
                new int[]{2,3,4,5,6,7,8},30,200000,.5,1,work,false,true,false,2,true);
            ArchiveJavaCollection.Result b=ArchiveJavaCollection.discover(reversed,"CT-ID",Paths.get("."),
                new int[]{2,3,4,5,6,7,8},30,200000,.5,1,work,false,true,false,2,true);
            if (a.selected.size()!=b.selected.size()) throw new AssertionError("Order-dependent dictionary size");
            for (int i=0;i<a.selected.size();i++) {
                ArchiveJavaCollection.Entry x=a.selected.get(i),y=b.selected.get(i);
                if (!x.pattern.equals(y.pattern) || x.support!=y.support)
                    throw new AssertionError("Order-dependent canonical ranking");
                for (int j=0;j<corpus.length;j++)
                    if (x.values[j]!=y.values[corpus.length-1-j]) throw new AssertionError("Reordered feature rows differ");
            }
            System.out.println("CT-ID audit passed: five engines, all pair counts, ties, skipped lengths, top-k/all output and sample-order invariance");
        } finally {
            Files.delete(work); // discover cleans up its own temporary spools.
        }
    }
}
