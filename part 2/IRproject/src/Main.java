import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


public class Main {

    public static void main(String[] args) {
        String filePath = "D:\\Downloads\\files\\MapReduceOutput.txt";
        Map<String, Map<Integer, Integer>> termDocumentPositions = readMapReduceOutput(filePath);
        Map<String, Map<Integer, Double>> tfTable = calculateTF(termDocumentPositions);
        Map<String, Map<Integer, Double>> WTFTable = calculateWTF(termDocumentPositions);
        Map<String, Double> idfTable = calculateIDF(termDocumentPositions);
        Map<String, Map<Integer, Double>> tfIdfTable = calculateTFIDF(tfTable, idfTable);
        Map<Integer, Double> documentLengths = calculateDocumentLengths(tfIdfTable);
        Map<String, Map<Integer, Double>> normalizedTfIdfTable = normalizeTFIDF(tfIdfTable, documentLengths);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1. Display Term Frequency Table");
            System.out.println("2. Display Weighted Term Frequency Table");
            System.out.println("3. Display IDF Table");
            System.out.println("4. Display TF-IDF Table");
            System.out.println("5. Display Document Lengths");
            System.out.println("6. Display Normalized TF-IDF Table");
            System.out.println("7. Perform Phrase Query and Rank Documents");
            System.out.println("8. Exit");
            System.out.print("Choose an option: ");

            int choice = -1;
            try {
                choice = scanner.nextInt();
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a number between 1 and 8.");
                scanner.nextLine();
                continue;
            }

            switch (choice) {
                case 1 -> printTable(tfTable, termDocumentPositions);
                case 2 -> printTable(WTFTable, termDocumentPositions);
                case 3 -> printIDFTable(idfTable, termDocumentPositions);
                case 4 -> printTable(tfIdfTable, termDocumentPositions);
                case 5 -> printDocumentLengths(documentLengths);
                case 6 -> printTable(normalizedTfIdfTable, termDocumentPositions);
                case 7 -> {
                    System.out.print("Enter your phrase query: ");
                    scanner.nextLine(); // Consume newline
                    String query = scanner.nextLine();
                    performPhraseQuery(query, termDocumentPositions, normalizedTfIdfTable, idfTable, documentLengths);
                }
                case 8 -> {
                    System.out.println("Exiting program.");
                    return;
                }
                default -> System.out.println("Invalid choice. Please enter a number between 1 and 8.");
            }
        }
    }

    private static Map<String, Map<Integer, Integer>> readMapReduceOutput(String filePath) {
        Map<String, Map<Integer, Integer>> termDocumentPositions = new TreeMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\\t");
                if (parts.length < 2 || parts[0].trim().isEmpty()) {
                    continue;
                }

                String term = parts[0].trim();
                String[] postings = parts[1].split("; ");
                Map<Integer, Integer> documentFrequency = new HashMap<>();

                for (String posting : postings) {
                    String[] docPos = posting.split(":");
                    if (docPos.length < 2 || docPos[0].trim().isEmpty() || docPos[1].trim().isEmpty()) {
                        continue;
                    }

                    try {
                        int docId = Integer.parseInt(docPos[0].trim());
                        String[] positions = docPos[1].split(",");

                        int frequency = positions.length;
                        documentFrequency.merge(docId, frequency, Integer::sum);
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid document ID or positions in line: " + line);
                    }
                }

                termDocumentPositions.put(term, documentFrequency);
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
        return termDocumentPositions;
    }

    private static Map<String, Map<Integer, Double>> calculateTF(Map<String, Map<Integer, Integer>> termDocumentPositions) {
        Map<String, Map<Integer, Double>> tfTable = new TreeMap<>();

        for (Map.Entry<String, Map<Integer, Integer>> termEntry : termDocumentPositions.entrySet()) {
            String term = termEntry.getKey();
            Map<Integer, Double> tfValues = new HashMap<>();

            for (Map.Entry<Integer, Integer> docEntry : termEntry.getValue().entrySet()) {
                int docId = docEntry.getKey();
                int frequency = docEntry.getValue();

                tfValues.put(docId, (double) frequency);
            }

            tfTable.put(term, tfValues);
        }

        return tfTable;
    }

    private static Map<String, Double> calculateIDF(Map<String, Map<Integer, Integer>> termDocumentPositions) {
        Map<String, Double> idfTable = new TreeMap<>();
        Set<Integer> allDocuments = new TreeSet<>();

        termDocumentPositions.values().forEach(map -> allDocuments.addAll(map.keySet()));
        int totalDocuments = allDocuments.size();

        for (Map.Entry<String, Map<Integer, Integer>> termEntry : termDocumentPositions.entrySet()) {
            int documentFrequency = termEntry.getValue().size();
            idfTable.put(termEntry.getKey(), Math.log10((double) totalDocuments / documentFrequency));
        }

        return idfTable;
    }

    private static Map<String, Map<Integer, Double>> calculateTFIDF(Map<String, Map<Integer, Double>> tfTable, Map<String, Double> idfTable) {
        Map<String, Map<Integer, Double>> tfIdfTable = new TreeMap<>();

        for (Map.Entry<String, Map<Integer, Double>> termEntry : tfTable.entrySet()) {
            String term = termEntry.getKey();
            double idf = idfTable.getOrDefault(term, 0.0);
            Map<Integer, Double> tfIdfValues = new HashMap<>();

            for (Map.Entry<Integer, Double> docEntry : termEntry.getValue().entrySet()) {
                tfIdfValues.put(docEntry.getKey(), docEntry.getValue() * idf);
            }

            tfIdfTable.put(term, tfIdfValues);
        }

        return tfIdfTable;
    }

    private static Map<Integer, Double> calculateDocumentLengths(Map<String, Map<Integer, Double>> tfIdfTable) {
        Map<Integer, Double> documentLengths = new HashMap<>();

        for (Map<Integer, Double> tfIdfValues : tfIdfTable.values()) {
            for (Map.Entry<Integer, Double> docEntry : tfIdfValues.entrySet()) {
                documentLengths.merge(docEntry.getKey(), Math.pow(docEntry.getValue(), 2), Double::sum);
            }
        }

        documentLengths.replaceAll((docId, length) -> Math.sqrt(length));
        return documentLengths;
    }

    private static Map<String, Map<Integer, Double>> normalizeTFIDF(Map<String, Map<Integer, Double>> tfIdfTable, Map<Integer, Double> documentLengths) {
        Map<String, Map<Integer, Double>> normalizedTable = new TreeMap<>();

        for (Map.Entry<String, Map<Integer, Double>> termEntry : tfIdfTable.entrySet()) {
            String term = termEntry.getKey();
            Map<Integer, Double> normalizedValues = new HashMap<>();

            for (Map.Entry<Integer, Double> docEntry : termEntry.getValue().entrySet()) {
                int docId = docEntry.getKey();
                double length = documentLengths.getOrDefault(docId, 1.0);
                normalizedValues.put(docId, docEntry.getValue() / length);
            }

            normalizedTable.put(term, normalizedValues);
        }

        return normalizedTable;
    }

    private static void printTable(Map<String, Map<Integer, Double>> table, Map<String, Map<Integer, Integer>> termDocumentPositions) {
        Set<Integer> allDocuments = new TreeSet<>();
        termDocumentPositions.values().forEach(map -> allDocuments.addAll(map.keySet()));

        System.out.printf("%-15s", "Term");
        for (int docId : allDocuments) {
            System.out.printf("D%-7d", docId);
        }
        System.out.println();

        for (Map.Entry<String, Map<Integer, Double>> termEntry : table.entrySet()) {
            System.out.printf("%-15s", termEntry.getKey());
            for (int docId : allDocuments) {
                System.out.printf("%-8.4f", termEntry.getValue().getOrDefault(docId, 0.0));
            }
            System.out.println();
        }
    }

    private static void printIDFTable(Map<String, Double> idfTable, Map<String, Map<Integer, Integer>> termDocumentPositions) {
        System.out.printf("%-15s%-15s%-15s\n", "Term", "DF", "IDF");
        for (Map.Entry<String, Double> idfEntry : idfTable.entrySet()) {
            String term = idfEntry.getKey();
            int documentFrequency = termDocumentPositions.get(term).size();
            System.out.printf("%-15s%-15d%-15.4f\n", term, documentFrequency, idfEntry.getValue());
        }
    }

    private static void printDocumentLengths(Map<Integer, Double> documentLengths) {
        documentLengths.forEach((docId, length) -> System.out.printf("Document %d Length: %.4f\n", docId, length));
    }

    private static Map<String, Map<Integer, Double>> calculateWTF(Map<String, Map<Integer, Integer>> termDocumentPositions) {
        Map<String, Map<Integer, Double>> tfTable = new TreeMap<>();

        for (Map.Entry<String, Map<Integer, Integer>> termEntry : termDocumentPositions.entrySet()) {
            String term = termEntry.getKey();
            Map<Integer, Double> tfValues = new HashMap<>();

            for (Map.Entry<Integer, Integer> docEntry : termEntry.getValue().entrySet()) {
                int docId = docEntry.getKey();
                int frequency = docEntry.getValue();


                tfValues.put(docId, (double) 1 + Math.log10(frequency));
            }

            tfTable.put(term, tfValues);
        }

        return tfTable;
    }

    private static void performPhraseQuery(String query, Map<String, Map<Integer, Integer>> termDocumentPositions,
                                           Map<String, Map<Integer, Double>> normalizedTfIdfTable, Map<String, Double> idfTable,
                                           Map<Integer, Double> documentLengths) {
        String[] terms = query.split(" ");
        Map<String, Double> queryVector = new HashMap<>();
        double queryLength = 0.0;

        System.out.printf("%-10s%-15s%-15s%-15s%-15s%-15s\n", "Term", "TF-raw", "WTF", "IDF", "TF*IDF", "Normalized");


        for (String term : terms) {
            int tfRaw = 1;
            double wtf = 1 + Math.log10(tfRaw);
            double idf = idfTable.getOrDefault(term, 0.0);
            double tfIdf = wtf * idf;
            queryVector.put(term, tfIdf);
            queryLength += Math.pow(tfIdf, 2);

            System.out.printf("%-10s%-15d%-15.4f%-15.4f%-15.4f\n", term, tfRaw, wtf, idf, tfIdf);
        }

        queryLength = Math.sqrt(queryLength);
        System.out.printf("\nQuery Length: %.6f\n\n", queryLength);


        double finalQueryLength = queryLength;
        queryVector.replaceAll((term, value) -> value / finalQueryLength);

        Set<Integer> relevantDocs = null;
        for (String term : terms) {
            if (termDocumentPositions.containsKey(term)) {
                Set<Integer> docs = termDocumentPositions.get(term).keySet();
                if (relevantDocs == null) {
                    relevantDocs = new HashSet<>(docs);
                } else {
                    relevantDocs.retainAll(docs);
                }
            } else {
                relevantDocs = Set.of();
                break;
            }
        }

        if (relevantDocs == null || relevantDocs.isEmpty()) {
            System.out.println("No matching documents found.");
            return;
        }

        Map<Integer, Double> similarityScores = new HashMap<>();
        for (int docId : relevantDocs) {
            double score = 0.0;
            for (String term : terms) {
                Map<Integer, Double> docVector = normalizedTfIdfTable.getOrDefault(term, Map.of());
                score += queryVector.getOrDefault(term, 0.0) * docVector.getOrDefault(docId, 0.0);
            }
            similarityScores.put(docId, score);
        }

        System.out.printf("%-10s%-15s\n", "Doc ID", "Similarity");
        similarityScores.entrySet().stream()
                .sorted((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()))
                .forEach(entry -> {
                    int docId = entry.getKey();
                    double score = entry.getValue();
                    if (score > 0) {
                        System.out.printf("%-10d%-15.4f\n", docId, score);
                    }
                });
    }
}