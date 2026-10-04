import java.util.*;

/** Deterministic preprocessing/grouping for ranked and pairwise paths. */
final class CTIntegerTools {
    private CTIntegerTools() {}

    /** Eight stable byte-radix passes over packed nonnegative 32-bit fields. */
    static int[] order(long[] keys,int size) {
        int[] order=new int[size], scratch=new int[size], buckets=new int[256];
        for (int i=0;i<size;i++) order[i]=i;
        for (int shift=0;shift<64;shift+=8) {
            Arrays.fill(buckets,0);
            for (int i:order) buckets[(int)(keys[i]>>>shift)&255]++;
            int offset=0;
            for (int b=0;b<256;b++) { int count=buckets[b]; buckets[b]=offset; offset+=count; }
            for (int i:order) scratch[buckets[(int)(keys[i]>>>shift)&255]++]=i;
            int[] swap=order; order=scratch; scratch=swap;
        }
        return order;
    }

    /** O(n log n) comparison sorting, no hash table; signed zeros are equal. */
    static int[] distinctLimits(double[] x) {
        Integer[] order=new Integer[x.length];
        int[] next=new int[x.length], limits=new int[x.length]; Arrays.fill(next,x.length);
        for (int i=0;i<x.length;i++) order[i]=i;
        Arrays.sort(order,(a,b)-> {
            double u=x[a]==0.?0.:x[a],v=x[b]==0.?0.:x[b];
            int c=Double.compare(u,v); return c==0 ? Integer.compare(a,b) : c;
        });
        for (int i=1;i<order.length;i++)
            if (x[order[i-1]]==x[order[i]]) next[order[i-1]]=order[i];
        int end=x.length;
        for (int i=x.length-1;i>=0;i--) { end=Math.min(end,next[i]); limits[i]=end-i; }
        return limits;
    }

    static int[] pd(double[] x) {
        int[] pd=new int[x.length], stack=new int[x.length]; int top=0;
        for (int i=0;i<x.length;i++) {
            if (!Double.isFinite(x[i])) throw new IllegalArgumentException("Nonfinite input");
            while (top>0 && x[stack[top-1]]>x[i]) top--;
            pd[i]=top==0 ? CTMiner.INF : i-stack[top-1]; stack[top++]=i;
        }
        return pd;
    }
}
