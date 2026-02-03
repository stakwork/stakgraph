import axios from "axios";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = "http://localhost:3355";
const BENCHMARK_FILE = path.join(
  __dirname,
  "../../benchmarks/search_queries.json",
);
const RESULTS_FILE = path.join(
  __dirname,
  "../../benchmarks/benchmark_results_vector.csv",
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
  node_types?: string[];
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
  category: string;
  query: string;
  node_types: string;
  k: number;
  expected_count: number;
  found_count: number;
  ranks: string;
  latency_ms: number;
  recall_at_k: number;
  top_result_name: string;
  top_result_type: string;
  top_result_file: string;
  found_names: string;
}

async function runVectorBenchmark(
  queries: QueryTestCase[],
): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  console.log(`\n=== Running VECTOR search benchmark ===\n`);
  // Start with a threshold of 0.6 as discussed
  const SIMILARITY_THRESHOLD = 0.6;
  console.log(`Similarity Threshold: ${SIMILARITY_THRESHOLD}`);

  for (const testCase of queries) {
    console.log(`Running query: "${testCase.query}" (ID: ${testCase.id})`);

    const start = Date.now();
    let responseData: any = null;

    try {
      const params: any = {
        query: testCase.query,
        limit: testCase.k,
        output: "json",
        method: "vector",
        similarityThreshold: SIMILARITY_THRESHOLD,
      };

      if (testCase.node_types && testCase.node_types.length > 0) {
        params.node_types = testCase.node_types.join(",");
      }

      const response = await axios.get(`${BASE_URL}/search`, { params });
      responseData = response.data;
    } catch (error: any) {
      console.error(`  Error: ${error.message}`);
      // Don't continue, push a failed result instead
      responseData = [];
    }

    const latency = Date.now() - start;
    const foundNodes: SearchResultNode[] = Array.isArray(responseData)
      ? responseData
      : responseData.nodes || [];

    const ranks: number[] = [];
    let hits = 0;

    for (const expected of testCase.expected.nodes) {
      const index = foundNodes.findIndex((node) => {
        const nameMatch = node.properties.name === expected.name;
        const typeMatch = node.node_type === expected.type;
        const fileMatch = expected.file
          ? node.properties.file.endsWith(expected.file)
          : true;
        return nameMatch && typeMatch && fileMatch;
      });

      if (index !== -1) {
        hits++;
        ranks.push(index + 1);
      } else {
        ranks.push(-1);
      }
    }

    const recall =
      testCase.expected.nodes.length > 0
        ? hits / testCase.expected.nodes.length
        : 0;

    const topResult = foundNodes.length > 0 ? foundNodes[0] : null;
    const topResultName = topResult ? topResult.properties.name : "";
    const topResultType = topResult ? topResult.node_type : "";
    const topResultFile = topResult ? topResult.properties.file : "";

    const foundNames = foundNodes
      .slice(0, 5)
      .map((n) => `${n.properties.name} (${n.node_type})`)
      .join("; ");

    const result: BenchmarkResult = {
      query_id: testCase.id,
      category: testCase.category,
      query: testCase.query,
      node_types: testCase.node_types ? testCase.node_types.join(",") : "",
      k: testCase.k,
      expected_count: testCase.expected.nodes.length,
      found_count: hits,
      ranks: ranks.join(","),
      latency_ms: latency,
      recall_at_k: recall,
      top_result_name: topResultName,
      top_result_type: topResultType,
      top_result_file: topResultFile,
      found_names: foundNames,
    };

    results.push(result);

    const status = recall === 1 ? "✓" : "✗";
    console.log(
      `  ${status} Hits: ${hits}/${testCase.expected.nodes.length}, Latency: ${latency}ms`,
    );
  }

  return results;
}

function writeResultsToCSV(results: BenchmarkResult[]) {
  const csvHeader =
    "query_id,category,query,node_types,k,expected_count,found_count,ranks,latency_ms,recall_at_k,top_result_name,top_result_type,top_result_file,found_names";

  const lines: string[] = [csvHeader];

  for (const result of results) {
    lines.push(formatResultLine(result));
  }

  fs.writeFileSync(RESULTS_FILE, lines.join("\n") + "\n");
  console.log(`\nResults written to: ${RESULTS_FILE}`);
}

function formatResultLine(result: BenchmarkResult): string {
  return [
    result.query_id,
    `"${result.category}"`,
    `"${result.query.replace(/"/g, '""')}"`,
    `"${result.node_types}"`,
    result.k,
    result.expected_count,
    result.found_count,
    `"${result.ranks}"`,
    result.latency_ms,
    result.recall_at_k.toFixed(2),
    `"${result.top_result_name.replace(/"/g, '""')}"`,
    `"${result.top_result_type}"`,
    `"${result.top_result_file.replace(/"/g, '""')}"`,
    `"${result.found_names.replace(/"/g, '""')}"`,
  ].join(",");
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

  console.log(`Loaded ${queries.length} queries`);

  const results = await runVectorBenchmark(queries);

  writeResultsToCSV(results);

  console.log("\n=== SUMMARY ===");

  const avgRecall =
    results.reduce((sum, r) => sum + r.recall_at_k, 0) / results.length;
  const perfectCount = results.filter((r) => r.recall_at_k === 1).length;
  const failedCount = results.filter((r) => r.recall_at_k === 0).length;

  console.log(`Average Recall: ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`Perfect Matches: ${perfectCount}/${results.length}`);
  console.log(`Failed: ${failedCount}/${results.length}`);

  console.log("\n--- Results by Category ---");
  const categories = [...new Set(results.map((r) => r.category))];
  for (const cat of categories) {
    const catResults = results.filter((r) => r.category === cat);
    const catRecall =
      catResults.reduce((sum, r) => sum + r.recall_at_k, 0) / catResults.length;
    const catPerfect = catResults.filter((r) => r.recall_at_k === 1).length;
    console.log(
      `  ${cat}: ${(catRecall * 100).toFixed(0)}% (${catPerfect}/${catResults.length})`,
    );
  }

  console.log("\n--- All Results ---");
  console.table(results);
}

runBenchmark().catch(console.error);
