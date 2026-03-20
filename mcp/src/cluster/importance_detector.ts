import { db } from "../graph/neo4j.js";

const GDS_GRAPH_NAME = "stakgraph_importance";

export interface ImportanceResult {
  nodesScored: number;
  topNodes: {
    ref_id: string;
    name: string;
    file: string;
    label: string;
    pagerank: number;
    in_degree: number;
    out_degree: number;
    entry_score: number;
    utility_score: number;
    hub_score: number;
  }[];
}

export async function runImportanceScoring(): Promise<ImportanceResult> {
  let graphProjected = false;
  try {
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

    const batch = degreeRows.map((row) => {
      const pagerank = pagerankMap.get(row.ref_id) ?? 0;
      const { in_degree, out_degree } = row;
      return {
        ref_id: row.ref_id,
        pagerank,
        in_degree,
        out_degree,
        entry_score: out_degree / (in_degree + 1),
        utility_score: in_degree / (out_degree + 1),
        hub_score: in_degree * out_degree,
      };
    });

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
