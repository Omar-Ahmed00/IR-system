
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.*;
import java.util.*;
import java.util.regex.*;
public class Mavenproject4 {

    public static void main(String[] args) {
        String filePath ="C:/Users/DELL-MCC/Desktop/MapReduceOutPut (1).txt";

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            Map<String, Set<Integer>> termPresenceMap = new HashMap<>();
            Map<Integer, Map<String, Integer>> termFrequencies = new HashMap<>();
            Map<Integer, Integer> documentTermCounts = new HashMap<>();

            // Calculate the total number of documents dynamically based on the input file
            int totalDocuments = 0;

            // Parse input file
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split("\\s+", 2);
                String term = parts[0];
                String postings = parts[1];

                Set<Integer> documentSet = new HashSet<>();
                Matcher matcher = Pattern.compile("(\\d+):(\\d+)").matcher(postings);

                while (matcher.find()) {
                    int docId = Integer.parseInt(matcher.group(1));
                    int frequency = Integer.parseInt(matcher.group(2));

                    documentSet.add(docId);

                    termFrequencies.putIfAbsent(docId, new HashMap<>());
                    termFrequencies.get(docId).put(term, frequency);

                    documentTermCounts.put(docId, documentTermCounts.getOrDefault(docId, 0) + frequency);

                    // Update totalDocuments based on the highest docId encountered
                    totalDocuments = Math.max(totalDocuments, docId);
                }

                termPresenceMap.put(term, documentSet);
            }

            // Sort termPresenceMap
            TreeMap<String, Set<Integer>> sortedTermPresenceMap = new TreeMap<>(termPresenceMap);

            // Compute IDF
            Map<String, Double> idfMap = computeIDF(sortedTermPresenceMap, totalDocuments);
            TreeMap<String, Double> sortedIDFMap = new TreeMap<>(idfMap);

            // Compute TF-IDF
            Map<String, Map<Integer, Double>> tfidfMap = computeTFIDF(sortedTermPresenceMap, idfMap, termFrequencies, documentTermCounts);

            // Display tables and process phrase queries based on user choice
            Scanner scanner = new Scanner(System.in);
            boolean keepGoing = true;

            while (keepGoing) {
                System.out.println("\nMenu:");
                System.out.println("1. Display Presence Table");
                System.out.println("2. Display IDF Table");
                System.out.println("3. Display TF-IDF Table");
                System.out.println("4. Enter Phrase Queries");
                System.out.println("5. Exit");
                System.out.print("Enter your choice: ");
                int choice = scanner.nextInt();
                scanner.nextLine(); // Consume newline

                switch (choice) {
                    case 1:
                        displayPresenceTable(sortedTermPresenceMap, totalDocuments);
                        break;
                    case 2:
                        displayIDFTable(sortedIDFMap);
                        break;
                    case 3:
                        displayTFIDFTable(tfidfMap, totalDocuments);
                        break;
                    case 4:
                        System.out.println("Enter phrase queries (type 'exit' to quit):");
                        while (true) {
                            System.out.print("> ");
                            String query = scanner.nextLine().trim();
                            if (query.equalsIgnoreCase("exit")) break;

                            Set<Integer> resultDocs = processQuery(query, termPresenceMap);
                            Map<Integer, Double> similarityScores = computeSimilarityScores(query, resultDocs, tfidfMap);
                            displayQueryResults(resultDocs, similarityScores);
                        }
                        break;
                    case 5:
                        keepGoing = false;
                        break;
                    default:
                        System.out.println("Invalid choice. Please try again.");
                }

                if (choice != 5) {
                    System.out.print("Do you want to make another choice? (yes/no): ");
                    String response = scanner.nextLine().trim().toLowerCase();
                    if (!response.equals("yes")) {
                        keepGoing = false;
                    }
                }
            }

        } catch (IOException e) {
            System.err.println("Error reading the file: " + e.getMessage());
        }
    }

    // Display Presence Table
    private static void displayPresenceTable(Map<String, Set<Integer>> termPresenceMap, int totalDocuments) {
        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print(" Doc" + i + "   |");
        }
        System.out.println();
        System.out.print("| Term           |");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print(" Presence |");
        }
        System.out.println();

        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print("----------+");
        }
        System.out.println();

        for (Map.Entry<String, Set<Integer>> entry : termPresenceMap.entrySet()) {
            String term = entry.getKey();
            Set<Integer> documentSet = entry.getValue();

            System.out.printf("| %-14s |", term);
            for (int i = 1; i <= totalDocuments; i++) {
                System.out.printf(" %-8d |", documentSet.contains(i) ? 1 : 0);
            }
            System.out.println();
        }

        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print("----------+");
        }
        System.out.println();
    }

    // Compute IDF
    private static Map<String, Double> computeIDF(Map<String, Set<Integer>> termPresenceMap, int totalDocuments) {
        Map<String, Double> idfMap = new HashMap<>();

        for (Map.Entry<String, Set<Integer>> entry : termPresenceMap.entrySet()) {
            String term = entry.getKey();
            int documentCount = entry.getValue().size();
            double idf = Math.log10((double) totalDocuments / documentCount);
            idfMap.put(term, idf);
        }

        return idfMap;
    }

    // Display IDF Table
    private static void displayIDFTable(Map<String, Double> idfMap) {
        System.out.println("\nIDF Values Table:");
        System.out.println("+----------------+------------+");
        System.out.println("| Term           | IDF Value  |");
        System.out.println("+----------------+------------+");

        for (Map.Entry<String, Double> entry : idfMap.entrySet()) {
            String term = entry.getKey();
            double idfValue = entry.getValue();

            System.out.printf("| %-14s | %-10.6f |\n", term, idfValue);
        }

        System.out.println("+----------------+------------+");
    }

    // Compute TF-IDF
    private static Map<String, Map<Integer, Double>> computeTFIDF(
            Map<String, Set<Integer>> termPresenceMap,
            Map<String, Double> idfMap,
            Map<Integer, Map<String, Integer>> termFrequencies,
            Map<Integer, Integer> documentTermCounts) {

        Map<String, Map<Integer, Double>> tfidfMap = new HashMap<>();

        for (String term : termPresenceMap.keySet()) {
            Map<Integer, Double> docTFIDFMap = new HashMap<>();
            for (int docId : termPresenceMap.get(term)) {
                int termFrequency = termFrequencies.get(docId).getOrDefault(term, 0);

                double tfidf = termFrequency * idfMap.get(term);
                docTFIDFMap.put(docId, tfidf);
            }
            tfidfMap.put(term, docTFIDFMap);
        }

        return tfidfMap;
    }

    // Display TF-IDF Table
    private static void displayTFIDFTable(Map<String, Map<Integer, Double>> tfidfMap, int totalDocuments) {
        System.out.println("\nTF-IDF Values Table:");
        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print(" Doc" + i + "         |");
        }
        System.out.println();

        System.out.print("| Term           |");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print(" TF-IDF Value    |");
        }
        System.out.println();

        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print("---------------+");
        }
        System.out.println();

        for (Map.Entry<String, Map<Integer, Double>> entry : tfidfMap.entrySet()) {
            String term = entry.getKey();
            Map<Integer, Double> docTFIDFMap = entry.getValue();

            System.out.printf("| %-14s |", term);
            for (int i = 1; i <= totalDocuments; i++) {
                Double tfidf = docTFIDFMap.get(i);
                System.out.printf(" %-14.6f |", tfidf != null ? tfidf : 0.0);
            }
            System.out.println();
        }

        System.out.print("+----------------+");
        for (int i = 1; i <= totalDocuments; i++) {
            System.out.print("---------------+");
        }
        System.out.println();
    }

    // Process query (can add custom logic for phrase processing)
    private static Set<Integer> processQuery(String query, Map<String, Set<Integer>> termPresenceMap) {
        Set<Integer> resultDocs = new HashSet<>();
        String[] terms = query.split("\\s+");

        for (String term : terms) {
            Set<Integer> docsWithTerm = termPresenceMap.get(term);
            if (docsWithTerm != null) {
                resultDocs.addAll(docsWithTerm);
            }
        }

        return resultDocs;
    }

    // Compute similarity scores (basic placeholder logic for demonstration)
    private static Map<Integer, Double> computeSimilarityScores(String query, Set<Integer> resultDocs, Map<String, Map<Integer, Double>> tfidfMap) {
        Map<Integer, Double> similarityScores = new HashMap<>();

        for (Integer docId : resultDocs) {
            double score = 0.0;

            // Example: Aggregate TF-IDF values (can be replaced with cosine similarity)
            for (String term : query.split("\\s+")) {
                Map<Integer, Double> tfidfValues = tfidfMap.get(term);
                if (tfidfValues != null && tfidfValues.containsKey(docId)) {
                    score += tfidfValues.get(docId);
                }
            }

            similarityScores.put(docId, score);
        }

        return similarityScores;
    }

    // Display query results
    private static void displayQueryResults(Set<Integer> resultDocs, Map<Integer, Double> similarityScores) {

        for (Integer docId : resultDocs) {
            double score = similarityScores.getOrDefault(docId, 0.0);
            System.out.printf("| %-6d | %-15.6f |\n", docId, score);
        }

        System.out.println("+--------+-----------------+");
        System.out.println("\nQuery Results:");
        System.out.println("+--------+-----------------+");
        System.out.println("| Doc ID | Similarity Score |");
        System.out.println("+--------+-----------------+");

        for (Integer docId : resultDocs) {
            double score = similarityScores.getOrDefault(docId, 0.0);
            System.out.printf("| %-6d | %-15.6f |\n", docId, score);
        }

        System.out.println("+--------+-----------------+");
    }
}