import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class PositionalIndexMapper extends Mapper<Object, Text, Text, Text> {
    private Text word = new Text();
    private Text docAndPos = new Text();

    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
        String docId = fileName.replace(".txt", ""); 

        String[] tokens = value.toString().split("\\s+"); // Split on whitespace
        for (int i = 0; i < tokens.length; i++) {
            word.set(tokens[i]);
            docAndPos.set(docId + ":" + (i + 1)); // Add 1 to make position 1-based
            context.write(word, docAndPos);
        }
    }
}