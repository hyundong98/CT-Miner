
import java.net.URLClassLoader;
import java.nio.file.*;
import java.io.*;
import java.util.*;
public class ReproBatch {
    // Class-loader isolation resets original static globals while amortizing
    // JVM startup over a collection. No forked subprocess or shell invocation.
    public static void main(String[] args) throws Exception {
        Path classes = Paths.get(args[0]);
        for (String line : Files.readAllLines(Paths.get(args[1]))) {
            String[] fields = line.split("\t", -1);
            try (URLClassLoader loader = new URLClassLoader(new java.net.URL[]{classes.toUri().toURL()}, ClassLoader.getPlatformClassLoader());
                 PrintStream output = new PrintStream(fields[1], "UTF-8")) {
                PrintStream previous = System.out;
                try {
                    System.setOut(output);
                    Class<?> c = Class.forName(fields[0], true, loader);
                    c.getMethod("main", String[].class).invoke(null, (Object)Arrays.copyOfRange(fields, 2, fields.length));
                } finally { System.setOut(previous); }
            }
        }
    }
}
