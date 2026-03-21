import GraphModule from 'graphology';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { db } from '../graph/neo4j.js';
import { generateLabel, detectLang } from "./lang/index.js";

// graphology is CJS; under Node16 module resolution the default import
// is the namespace object, not the constructor directly.
const Graph = (GraphModule as any).default ?? GraphModule;
type GraphInstance = InstanceType<typeof Graph>;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const leidenPath = resolve(__dirname, '..', '..', 'vendor', 'leiden', 'index.cjs');
const _require = createRequire(import.meta.url);
const leiden = _require(leidenPath);

export interface ClusterDetectionResult {
  clusterCount: number;
  modularity: number;
  nodesProcessed: number;
}

export async function runClusterDetection(): Promise<ClusterDetectionResult> {
  const { nodes, edges } = await db.get_cluster_graph_data();
  console.log(`[clusters] ${nodes.length} nodes, ${edges.length} edges`);

  if (nodes.length === 0) {
    return { clusterCount: 0, modularity: 0, nodesProcessed: 0 };
  }

  const graph = new Graph({ type: 'undirected', allowSelfLoops: false });

  const nodeSet = new Set<string>();
  for (const n of nodes) {
    if (n.ref_id && !nodeSet.has(n.ref_id)) {
      nodeSet.add(n.ref_id);
      graph.addNode(n.ref_id, { name: n.name, file: n.file, label: n.label });
    }
  }
  for (const e of edges) {
    if (graph.hasNode(e.source) && graph.hasNode(e.target) && e.source !== e.target) {
      if (!graph.hasEdge(e.source, e.target)) {
        graph.addEdge(e.source, e.target);
      }
    }
  }

  const resolution = graph.order >= 10000 ? 1.5 : 0.7;
  const details = leiden.detailed(graph, { resolution, randomWalk: true });
  console.log(`[clusters] modularity=${details.modularity.toFixed(3)}`);

  const communityMembers = new Map<number, string[]>();
  for (const [nodeId, commNum] of Object.entries(details.communities as Record<string, number>)) {
    if (!communityMembers.has(commNum)) communityMembers.set(commNum, []);
    communityMembers.get(commNum)!.push(nodeId);
  }

  const allFiles = nodes.map((n) => n.file || "");
  const lang = detectLang(allFiles);
  const usedLabels = new Set<string>();

  const MERGE_SIZE_THRESHOLD = 10;
  type CommunityEntry = { commNum: number; memberIds: string[]; label: string };

  const qualified: CommunityEntry[] = [];
  for (const [commNum, memberIds] of communityMembers) {
    if (memberIds.length < 3) continue;
    const files = memberIds.map(
      (id) => (graph.getNodeAttribute(id, "file") as string) || "",
    );
    qualified.push({
      commNum,
      memberIds,
      label: generateLabel(files, commNum, usedLabels, lang),
    });
  }

  // Merge small clusters that share the same label
  const labelGroups = new Map<string, CommunityEntry[]>();
  for (const entry of qualified) {
    if (!labelGroups.has(entry.label)) labelGroups.set(entry.label, []);
    labelGroups.get(entry.label)!.push(entry);
  }

  const finalClusters: { clusterId: string; label: string; memberIds: string[] }[] = [];
  for (const [label, group] of labelGroups) {
    const small = group.filter(e => e.memberIds.length <= MERGE_SIZE_THRESHOLD);
    const large = group.filter(e => e.memberIds.length > MERGE_SIZE_THRESHOLD);

    for (const entry of large) {
      finalClusters.push({ clusterId: `cluster_${entry.commNum}`, label, memberIds: entry.memberIds });
    }
    if (small.length > 0) {
      const merged = small.flatMap(e => e.memberIds);
      finalClusters.push({ clusterId: `cluster_${small[0].commNum}`, label, memberIds: merged });
    }
  }

  await db.clear_clusters();

  const clusterRows = finalClusters.map(({ clusterId, label, memberIds }) => ({
    cluster_id: clusterId,
    label,
    cohesion: calculateCohesion(memberIds, graph),
    symbol_count: memberIds.length,
  }));
  await db.bulk_upsert_clusters(clusterRows);

  const memberEdges = finalClusters.flatMap(({ clusterId, memberIds }) =>
    memberIds.map((refId) => ({ ref_id: refId, cluster_id: clusterId })),
  );
  await db.bulk_create_member_of(memberEdges);

  console.log(`[clusters] done: ${finalClusters.length} clusters, ${graph.order} nodes`);
  return {
    clusterCount: finalClusters.length,
    modularity: details.modularity,
    nodesProcessed: graph.order,
  };
}

function calculateCohesion(memberIds: string[], graph: GraphInstance): number {
  if (memberIds.length <= 1) return 1.0;
  const memberSet = new Set(memberIds);
  let internal = 0;
  let total = 0;
  for (const nodeId of memberIds) {
    if (graph.hasNode(nodeId)) {
      graph.forEachNeighbor(nodeId, (neighbor: string) => {
        total++;
        if (memberSet.has(neighbor)) internal++;
      });
    }
  }
  return total === 0 ? 1.0 : Math.min(1.0, internal / total);
}
