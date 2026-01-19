import fs from "fs";
import path from "path";

// Config
const API_URL = "http://localhost:3355/search";

const BENCHMARKS_DIR = path.join(process.cwd(), "..", "benchmarks");
const BENCHMARK_FILE = path.join(BENCHMARKS_DIR, "search_queries.json");
const OUTPUT_CSV = path.join(BENCHMARKS_DIR, "benchmark_results.csv");

// Types
interface Expectation {
  file_endswith?: string;
  name?: string;
  node_type?: string;
}

interface QueryCase {
  category: string;
  query: string;
  k: number;
  expected: Expectation[];
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
  category: string;
  query: string;
  expected_desc: string;
  actual_desc: string;
  found: boolean;
  rank: number | string;
  latency_ms: number;
}

// Helpers
function matchesExpectation(
  node: SearchResultNode,
  expected: Expectation,
): boolean {
  if (
    expected.file_endswith &&
    !node.properties.file?.endsWith(expected.file_endswith)
  ) {
    return false;
  }
  if (expected.name && node.properties.name !== expected.name) {
    return false;
  }
  if (expected.node_type && node.node_type !== expected.node_type) {
    return false;
  }
  return true;
}

// Main
async function runBenchmark() {
  console.log(`Reading benchmark queries from: ${BENCHMARK_FILE}`);

  let queries: QueryCase[] = [];
  try {
    if (!fs.existsSync(BENCHMARK_FILE)) {
      console.error(`Benchmark file not found at: ${BENCHMARK_FILE}`);
      console.error(`CWD: ${process.cwd()}`);
      process.exit(1);
    }
    const data = fs.readFileSync(BENCHMARK_FILE, "utf-8");
    queries = JSON.parse(data);
  } catch (e) {
    console.error(`Failed to read benchmark file: ${e}`);
    process.exit(1);
  }

  console.log(`Found ${queries.length} queries to run.`);
  const results: BenchmarkResult[] = [];

  for (const q of queries) {
    process.stdout.write(`Benchmarking: "${q.query}" ... `);

    const start = performance.now();
    let responseNodes: SearchResultNode[] = [];
    let fetchError = false;

    try {
      const params = new URLSearchParams({
        query: q.query,
        limit: q.k.toString(),
        output: "json",
      });

      const res = await fetch(`${API_URL}?${params}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      responseNodes = await res.json();
    } catch (e) {
      console.error(`Error fetching query "${q.query}":`, e);
      fetchError = true;
    }

    const latency = performance.now() - start;

    let found = false;
    let rank: number | string = "-";
    let topResultsDesc = "No results";

    if (!fetchError && Array.isArray(responseNodes)) {
      // Capture top 5 for reporting
      const topNodes = responseNodes.slice(0, 5);
      if (topNodes.length > 0) {
        topResultsDesc = topNodes
          .map(
            (n) =>
              `[${n.node_type}: ${n.properties.name || "unnamed"} (${n.properties.file || "no-file"})]`,
          )
          .join(" | ");
      }

      for (let i = 0; i < responseNodes.length; i++) {
        const node = responseNodes[i];
        const isMatch = q.expected.some((exp) => matchesExpectation(node, exp));
        if (isMatch) {
          found = true;
          rank = i + 1;
          break;
        }
      }
    }

    const resultEntry: BenchmarkResult = {
      category: q.category,
      query: q.query,
      expected_desc: q.expected
        .map((e) => `${e.node_type}:${e.name} (${e.file_endswith})`)
        .join(" OR "),
      actual_desc: topResultsDesc,
      found,
      rank,
      latency_ms: Math.round(latency),
    };

    results.push(resultEntry);
    console.log(found ? "FOUND" : "MISSED");
  }

  // Generate CSV
  console.log(`\nWriting results to: ${OUTPUT_CSV}`);
  const csvHeader =
    "Category,Query,Expected,Actual Top 5,Found,Rank,Latency(ms)\n";
  const csvRows = results
    .map((r) => {
      const safeQuery = `"${r.query.replace(/"/g, '""')}"`;
      const safeExpected = `"${r.expected_desc.replace(/"/g, '""')}"`;
      const safeActual = `"${r.actual_desc.replace(/"/g, '""')}"`;
      return `${r.category},${safeQuery},${safeExpected},${safeActual},${r.found},${r.rank},${r.latency_ms}`;
    })
    .join("\n");

  fs.writeFileSync(OUTPUT_CSV, csvHeader + csvRows);

  const passed = results.filter((r) => r.found).length;
  console.log(`\nBenchmark Complete.`);
  console.log(`Total: ${queries.length}`);
  console.log(`Found: ${passed}`);
  console.log(`Missed: ${queries.length - passed}`);
  console.log(`Accuracy: ${((passed / queries.length) * 100).toFixed(1)}%`);
}

runBenchmark();
