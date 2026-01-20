import axios from "axios";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const BASE_URL = "http://localhost:3355";
const BENCHMARK_FILE = path.join(
  __dirname,
  "../../benchmarks/search_queries.json",
);
const RESULTS_FILE = path.join(
  __dirname,
  "../../benchmarks/benchmark_results.csv",
);

interface ExpectedNode {
  name: string;
  type: string;
  file?: string;
}

interface QueryTestCase {
  id: string;
  query: string;
  category: string;
  k: number;
  node_types?: string[]; // Optional node type filter
  expected: {
    nodes: ExpectedNode[];
  };
}

interface SearchResultNode {
  node_type: string;
  properties: {
    name: string;
    file: string;
    [key: string]: any;
  };
}

interface BenchmarkResult {
  query_id: string;
  query: string;
  node_types: string; // Comma-separated or empty
  k: number;
  expected_count: number;
  found_count: number;
  ranks: string; // "1,5,-1" where -1 means not found
  latency_ms: number;
  recall_at_k: number;
}

async function runBenchmark() {
  console.log(`Loading benchmark queries from ${BENCHMARK_FILE}`);

  if (!fs.existsSync(BENCHMARK_FILE)) {
    console.error(`Benchmark file not found: ${BENCHMARK_FILE}`);
    process.exit(1);
  }

  const queries: QueryTestCase[] = JSON.parse(
    fs.readFileSync(BENCHMARK_FILE, "utf-8"),
  );
  const results: BenchmarkResult[] = [];

  // Header with more details
  const csvHeader =
    "query_id,query,node_types,k,expected_count,found_count,ranks,latency_ms,recall_at_k,top_result_name,top_result_type,top_result_file,found_names";
  fs.writeFileSync(RESULTS_FILE, csvHeader + "\n");
  console.log(`Initialized results file: ${RESULTS_FILE}`);

  for (const testCase of queries) {
    console.log(`Running query: "${testCase.query}" (ID: ${testCase.id})`);

    const start = Date.now();
    let responseData: any = null;

    try {
      const url = `${BASE_URL}/search`;
      const params: any = {
        query: testCase.query,
        limit: testCase.k,
        output: "json",
      };

      // Add node_types if specified
      if (testCase.node_types && testCase.node_types.length > 0) {
        params.node_types = testCase.node_types.join(",");
      }

      const response = await axios.get(url, { params });
      responseData = response.data;
    } catch (error: any) {
      console.error(
        `Error executing query "${testCase.query}":`,
        error.message,
      );
      continue;
    }

    const latency = Date.now() - start;
    const foundNodes: SearchResultNode[] = Array.isArray(responseData)
      ? responseData
      : responseData.nodes || [];

    // Evaluate results
    const ranks: number[] = [];
    let hits = 0;

    for (const expected of testCase.expected.nodes) {
      const index = foundNodes.findIndex((node) => {
        const nameMatch = node.properties.name === expected.name;
        // const typeMatch = node.node_type === expected.type;
        // loosening type check for now since we haven't confirmed exact mapping
        // const typeMatch = true;
        const typeMatch = node.node_type === expected.type;

        const fileMatch = expected.file
          ? node.properties.file.endsWith(expected.file)
          : true;

        return nameMatch && typeMatch && fileMatch;
      });

      if (index !== -1) {
        hits++;
        ranks.push(index + 1); // 1-based rank
      } else {
        ranks.push(-1);
      }
    }

    const recall =
      testCase.expected.nodes.length > 0
        ? hits / testCase.expected.nodes.length
        : 0;

    // Extract detailed result info
    const topResult = foundNodes.length > 0 ? foundNodes[0] : null;
    const topResultName = topResult ? topResult.properties.name : "";
    const topResultType = topResult ? topResult.node_type : "";
    const topResultFile = topResult ? topResult.properties.file : "";

    // Get first 5 names found
    const foundNames = foundNodes
      .slice(0, 5)
      .map((n) => `${n.properties.name} (${n.node_type})`)
      .join("; ");

    const result: BenchmarkResult = {
      query_id: testCase.id,
      query: testCase.query,
      node_types: testCase.node_types ? testCase.node_types.join(",") : "",
      k: testCase.k,
      expected_count: testCase.expected.nodes.length,
      found_count: hits,
      ranks: ranks.join(","),
      latency_ms: latency,
      recall_at_k: recall,
    };

    results.push(result);

    // Append to CSV
    const csvLine = [
      result.query_id,
      `"${result.query.replace(/"/g, '""')}"`,
      `"${result.node_types}"`,
      result.k,
      result.expected_count,
      result.found_count,
      `"${result.ranks}"`,
      result.latency_ms,
      result.recall_at_k.toFixed(2),
      `"${topResultName.replace(/"/g, '""')}"`,
      `"${topResultType}"`,
      `"${topResultFile.replace(/"/g, '""')}"`,
      `"${foundNames.replace(/"/g, '""')}"`,
    ].join(",");

    fs.appendFileSync(RESULTS_FILE, csvLine + "\n");

    console.log(
      `  Hits: ${hits}/${testCase.expected.nodes.length}, Latency: ${latency}ms, Ranks: [${ranks.join(", ")}]`,
    );
  }

  console.log("\nBenchmark completed.");
  console.table(results);
}

runBenchmark().catch(console.error);
