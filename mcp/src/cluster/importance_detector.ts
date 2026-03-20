import { db } from "../graph/neo4j.js";
import {
  ImportanceTag,
  ImportanceThresholds,
  ImportanceResult,
  ScoredNode,
  TaggedNode,
} from "./types.js";

const GDS_GRAPH_NAME = "stakgraph_importance";

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.floor((p / 100) * sorted.length);
  return sorted[Math.min(idx, sorted.length - 1)];
}

function assignTag(
  node_type: string,
  in_degree: number,
  out_degree: number,
  entry_score: number,
  utility_score: number,
  hub_score: number,
  thresholds: ImportanceThresholds,
): ImportanceTag {
  // Structural overrides — node types that are entry points by definition
  if (node_type === "Request" || node_type === "Page") {
    return ImportanceTag.EntryPoint;
  }
  // Endpoints with no callers in the projected graph are pure API entry points
  if (node_type === "Endpoint" && in_degree === 0) {
    return ImportanceTag.EntryPoint;
  }

  // Hub: significant traffic flowing both in AND out (top 10% of non-zero hub_score)
  if (hub_score > 0 && hub_score >= thresholds.hub_p90) {
    return ImportanceTag.Hub;
  }

  // Entry point: top 10% entry_score AND meaningfully calls out (≥3 callees)
  // Using p90 + min out_degree avoids tagging every function that calls one helper
  if (entry_score >= thresholds.entry_p90 && out_degree >= 3) {
    return ImportanceTag.EntryPoint;
  }

  // Utility: heavily called leaf nodes (top 25% utility_score, has callers)
  if (utility_score >= thresholds.utility_p75 && in_degree > 0) {
    return ImportanceTag.Utility;
  }

  // Connector: everything else — including nodes that appear "isolated" in the
  // projected graph (no CALLS/HANDLER/RENDERS edges) but may have IMPLEMENTS,
  // CONTAINS, etc. They are still part of the codebase, not truly isolated.
  return ImportanceTag.Connector;
}

export async function runImportanceScoring(): Promise<ImportanceResult> {
  let graphProjected = false;
  try {
    // Drop any leftover projection from a previously interrupted run
    await db.drop_importance_graph(GDS_GRAPH_NAME).catch(() => {});
    const nodeCount = await db.project_importance_graph(GDS_GRAPH_NAME);
    graphProjected = true;
    console.log(`[importance] projected graph with ${nodeCount} nodes`);

    const [pagerankRows, degreeRows] = await Promise.all([
      db.stream_pagerank(GDS_GRAPH_NAME),
      db.get_degree_counts(),
    ]);

    const pagerankMap = new Map<string, number>();
    for (const row of pagerankRows) {
      pagerankMap.set(row.ref_id, row.score);
    }

    // Compute raw scores first pass
    const scored: ScoredNode[] = degreeRows.map((row) => {
      const pagerank = pagerankMap.get(row.ref_id) ?? 0;
      const { in_degree, out_degree } = row;
      return {
        ref_id: row.ref_id,
        node_type: row.node_type,
        pagerank,
        in_degree,
        out_degree,
        entry_score: out_degree / (in_degree + 1),
        utility_score: in_degree / (out_degree + 1),
        hub_score: in_degree * out_degree,
      };
    });

    // Build percentile thresholds from non-structural nodes only
    const nonStructural = scored.filter(
      (n) =>
        n.node_type !== "Request" &&
        n.node_type !== "Page" &&
        !(n.node_type === "Endpoint" && n.in_degree === 0),
    );
    const sortedEntry = nonStructural
      .map((n) => n.entry_score)
      .sort((a, b) => a - b);
    const sortedUtility = nonStructural
      .map((n) => n.utility_score)
      .sort((a, b) => a - b);
    const sortedHub = nonStructural
      .map((n) => n.hub_score)
      .sort((a, b) => a - b);

    // Compute hub and entry thresholds on non-zero values only to avoid dilution
    const nonZeroHub = sortedHub.filter((v) => v > 0);
    const thresholds: ImportanceThresholds = {
      entry_p90: percentile(sortedEntry, 90),
      utility_p75: percentile(sortedUtility, 75),
      hub_p90: percentile(nonZeroHub, 90),
    };
    console.log(
      `[importance] thresholds: entry_p90=${thresholds.entry_p90.toFixed(3)} utility_p75=${thresholds.utility_p75.toFixed(3)} hub_p90=${thresholds.hub_p90}`,
    );

    // Assign tags
    const batch: TaggedNode[] = scored.map((n) => ({
      ref_id: n.ref_id,
      node_type: n.node_type,
      pagerank: n.pagerank,
      in_degree: n.in_degree,
      out_degree: n.out_degree,
      entry_score: n.entry_score,
      utility_score: n.utility_score,
      hub_score: n.hub_score,
      importance_tag: assignTag(
        n.node_type ?? "",
        n.in_degree,
        n.out_degree,
        n.entry_score,
        n.utility_score,
        n.hub_score,
        thresholds,
      ),
    }));

    // Log tag distribution
    const tagCounts = batch.reduce(
      (acc, n) => {
        acc[n.importance_tag] = (acc[n.importance_tag] ?? 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );
    console.log(`[importance] tag distribution:`, tagCounts);

    await db.bulk_update_importance(batch);
    console.log(`[importance] scored ${batch.length} nodes`);

    const topNodes = await db.get_top_nodes_by_importance(50);
    return { nodesScored: batch.length, topNodes };
  } finally {
    if (graphProjected) {
      await db.drop_importance_graph(GDS_GRAPH_NAME).catch(() => {});
    }
  }
}
