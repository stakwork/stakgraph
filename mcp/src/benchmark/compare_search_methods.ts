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
  "../../benchmarks/benchmark_comparison.csv",
);

// Reuse types from existing benchmarks
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

interface ComparisonResult {
  query_id: string;
  category: string;
  query: string;

  // Fulltext stats
  ft_found: boolean;
  ft_rank: number;
  ft_latency: number;
  ft_top_result: string;

  // Vector stats
  v_found: boolean;
  v_rank: number;
  v_latency: number;
  v_top_result: string;

  // Comparison
  winner:
    | "Fulltext"
    | "Vector"
    | "Tie (Both Found)"
    | "Tie (Both Missed)"
    | "Mixed";
  notes: string;
}

async function runSearch(
  query: string,
  method: "fulltext" | "vector",
  k: number,
  nodeTypes?: string[],
): Promise<{
  nodes: SearchResultNode[];
  latency: number;
}> {
  const start = Date.now();
  try {
    const params: any = {
      query,
      limit: k,
      output: "json",
      method,
    };

    if (method === "vector") {
      params.similarityThreshold = 0.6;
    }

    if (nodeTypes && nodeTypes.length > 0) {
      params.node_types = nodeTypes.join(",");
    }

    const response = await axios.get(`${BASE_URL}/search`, { params });
    const nodes = Array.isArray(response.data)
      ? response.data
      : response.data.nodes || [];
    return { nodes, latency: Date.now() - start };
  } catch (error: any) {
    // console.error(`Error in ${method} search for "${query}": ${error.message}`);
    return { nodes: [], latency: Date.now() - start };
  }
}

function checkHits(
  foundNodes: SearchResultNode[],
  expectedNodes: ExpectedNode[],
): { found: boolean; rank: number } {
  if (expectedNodes.length === 0) return { found: false, rank: -1 };

  // For benchmark, we consider it "Found" if ANY of the expected nodes appear in the results
  // We return the BEST rank of any hit
  let bestRank = -1;
  let isFound = false;

  for (const expected of expectedNodes) {
    const index = foundNodes.findIndex((node) => {
      const nameMatch = node.properties.name === expected.name;
      const typeMatch = node.node_type === expected.type;
      const fileMatch = expected.file
        ? node.properties.file.endsWith(expected.file)
        : true;
      return nameMatch && typeMatch && fileMatch;
    });

    if (index !== -1) {
      isFound = true;
      const rank = index + 1;
      if (bestRank === -1 || rank < bestRank) {
        bestRank = rank;
      }
    }
  }

  return { found: isFound, rank: bestRank };
}

async function runComparison() {
  console.log(`Loading benchmark queries from ${BENCHMARK_FILE}`);
  if (!fs.existsSync(BENCHMARK_FILE)) {
    console.error(`Benchmark file not found: ${BENCHMARK_FILE}`);
    process.exit(1);
  }

  const queries: QueryTestCase[] = JSON.parse(
    fs.readFileSync(BENCHMARK_FILE, "utf-8"),
  );
  console.log(`Loaded ${queries.length} queries. Running comparison...\n`);

  const results: ComparisonResult[] = [];

  for (const testCase of queries) {
    process.stdout.write(
      `Query: "${testCase.query.substring(0, 30)}${testCase.query.length > 30 ? "..." : ""}"... `,
    );

    // Run Fulltext
    const ftResult = await runSearch(
      testCase.query,
      "fulltext",
      testCase.k,
      testCase.node_types,
    );
    const ftCheck = checkHits(ftResult.nodes, testCase.expected.nodes);

    // Run Vector
    const vResult = await runSearch(
      testCase.query,
      "vector",
      testCase.k,
      testCase.node_types,
    );
    const vCheck = checkHits(vResult.nodes, testCase.expected.nodes);

    // Compare
    let winner: ComparisonResult["winner"] = "Tie (Both Missed)";
    if (ftCheck.found && vCheck.found) {
      winner = "Tie (Both Found)";
    } else if (ftCheck.found && !vCheck.found) {
      winner = "Fulltext";
    } else if (!ftCheck.found && vCheck.found) {
      winner = "Vector";
    }

    const formatTopResult = (nodes: SearchResultNode[]) => {
      if (nodes.length === 0) return "";
      const n = nodes[0];
      return `${n.properties.name} (${n.node_type}) - ${n.properties.file.split("/").pop()}`;
    };

    results.push({
      query_id: testCase.id,
      category: testCase.category,
      query: testCase.query,

      ft_found: ftCheck.found,
      ft_rank: ftCheck.rank,
      ft_latency: ftResult.latency,
      ft_top_result: formatTopResult(ftResult.nodes),

      v_found: vCheck.found,
      v_rank: vCheck.rank,
      v_latency: vResult.latency,
      v_top_result: formatTopResult(vResult.nodes),

      winner,
      notes: "",
    });

    console.log(
      `FT: ${ftCheck.found ? "✓" : "✗"} | V: ${vCheck.found ? "✓" : "✗"} -> Winner: ${winner}`,
    );
  }

  // Write CSV
  const csvHeader =
    "query_id,category,query,ft_found,v_found,winner,ft_rank,v_rank,ft_latency,v_latency,ft_top_result,v_top_result";
  const lines = results.map((r) =>
    [
      r.query_id,
      `"${r.category}"`,
      `"${r.query.replace(/"/g, '""')}"`,
      r.ft_found ? 1 : 0,
      r.v_found ? 1 : 0,
      r.winner,
      r.ft_rank,
      r.v_rank,
      r.ft_latency,
      r.v_latency,
      `"${r.ft_top_result.replace(/"/g, '""')}"`,
      `"${r.v_top_result.replace(/"/g, '""')}"`,
    ].join(","),
  );

  fs.writeFileSync(RESULTS_FILE, [csvHeader, ...lines].join("\n"));
  console.log(`\nComparison saved to: ${RESULTS_FILE}`);

  // Summary
  console.log("\n=== SUMMARY ===");
  const vectorWins = results.filter((r) => r.winner === "Vector").length;
  const ftWins = results.filter((r) => r.winner === "Fulltext").length;
  const bothFound = results.filter(
    (r) => r.winner === "Tie (Both Found)",
  ).length;
  const bothMissed = results.filter(
    (r) => r.winner === "Tie (Both Missed)",
  ).length;

  console.log(`Total Queries: ${results.length}`);
  console.log(`Fulltext Wins: ${ftWins} (Unique finds)`);
  console.log(`Vector Wins:   ${vectorWins} (Unique finds)`);
  console.log(`Both Found:    ${bothFound}`);
  console.log(`Both Missed:   ${bothMissed}`);

  console.log(`\nGlobal Recall:`);
  console.log(
    `Fulltext: ${(((ftWins + bothFound) / results.length) * 100).toFixed(1)}%`,
  );
  console.log(
    `Vector:   ${(((vectorWins + bothFound) / results.length) * 100).toFixed(1)}%`,
  );
  console.log(
    `Hybrid:   ${(((ftWins + vectorWins + bothFound) / results.length) * 100).toFixed(1)}% (Potential)`,
  );
}

runComparison().catch(console.error);
