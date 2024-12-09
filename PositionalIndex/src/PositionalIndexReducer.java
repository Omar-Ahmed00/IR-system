import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class PositionalIndexReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        Map<String, StringBuilder> positionMap = new HashMap<>();

        for (Text val : values) {
            String[] docAndPos = val.toString().split(":");
            String docId = docAndPos[0];
            String position = docAndPos[1];

//                       .append(position).append(",");
            if (!positionMap.containsKey(docId)) {
                positionMap.put(docId, new StringBuilder());
            }
            positionMap.get(docId).append(position).append(",");
        }

        StringBuilder output = new StringBuilder();
        for (Map.Entry<String, StringBuilder> entry : positionMap.entrySet()) {
            output.append(entry.getKey())
                  .append(":")
                  .append(entry.getValue().toString().replaceAll(",$", ""))
                  .append("; ");
        }

        context.write(key, new Text(output.toString().trim()));
    }
}