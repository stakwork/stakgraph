import GraphModule from 'graphology';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { db } from '../graph/neo4j.js';
import { ClusterDetectionResult } from './detector.js';

const Graph = (GraphModule as any).default ?? GraphModule;
type GraphInstance = InstanceType<typeof Graph>;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const leidenPath = resolve(__dirname, '..', '..', 'vendor', 'leiden', 'index.cjs');
const _require = createRequire(import.meta.url);
const leiden = _require(leidenPath);

const GDS_GRAPH_NAME = 'stakgraph_semantic';
const KNN_TOP_K = 10;
const KNN_SAMPLE_RATE = 0.5;
const KNN_SIMILARITY_CUTOFF = 0.62;
const MIN_COMMUNITY_SIZE = 3;

const GENERIC_FOLDERS = new Set([
  'src', 'lib', 'core', 'utils', 'common', 'shared', 'helpers',
  'queries', 'mutations', 'hooks', 'components', 'pages', 'views',
  'services', 'store', 'stores', 'api', 'types', 'interfaces',
]);

const GENERIC_FILENAMES = new Set([
  'route', 'page', 'index', 'layout', 'loading', 'error', 'not-found',
  'middleware', 'handler', 'mod', 'main', 'app',
]);

function generateSemanticLabel(members: { file: string }[], commNum: number): string {
  const counts = new Map<string, number>();
  for (const { file } of members) {
    const parts = file.split('/').filter(Boolean);
    for (let i = parts.length - 1; i >= 0; i--) {
      const raw = parts[i].replace(/\.[^.]+$/, '');
      const segment = raw.replace(/\.[^.]+$/, ''); // strip double ext e.g. .test.ts
      const lower = segment.toLowerCase();
      if (GENERIC_FILENAMES.has(lower) || GENERIC_FOLDERS.has(lower) || segment.length <= 1) {
        continue;
      }
      counts.set(segment, (counts.get(segment) || 0) + 1);
      break;
    }
  }
  if (counts.size > 0) {
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
    return best.charAt(0).toUpperCase() + best.slice(1);
  }
  return `SemanticCluster_${commNum}`;
}

export async function runSemanticClusterDetection(): Promise<ClusterDetectionResult> {
  try {
    const nodeCount = await db.project_semantic_graph(GDS_GRAPH_NAME);
    console.log(`[semantic-clusters] projected ${nodeCount} nodes`);

    if (nodeCount === 0) {
      return { clusterCount: 0, modularity: 0, nodesProcessed: 0 };
    }

    const pairs = await db.stream_semantic_knn(GDS_GRAPH_NAME, KNN_TOP_K, KNN_SAMPLE_RATE, KNN_SIMILARITY_CUTOFF);
    console.log(`[semantic-clusters] KNN returned ${pairs.length} similarity pairs`);

    // Build graphology graph from KNN pairs for vendored Leiden
    const graph: GraphInstance = new Graph({ type: 'undirected', allowSelfLoops: false });
    const nodeAttrs = new Map<string, { name: string; file: string }>();

    for (const p of pairs) {
      if (p.source && !nodeAttrs.has(p.source)) {
        nodeAttrs.set(p.source, { name: p.sourceName, file: p.sourceFile });
      }
    }
    for (const [ref_id, attrs] of nodeAttrs) {
      graph.addNode(ref_id, attrs);
    }
    for (const p of pairs) {
      if (p.source && p.target && graph.hasNode(p.source) && graph.hasNode(p.target) && p.source !== p.target) {
        if (!graph.hasEdge(p.source, p.target)) {
          graph.addEdge(p.source, p.target, { weight: p.similarity });
        }
      }
    }

    console.log(`[semantic-clusters] graph: ${graph.order} nodes, ${graph.size} edges`);

    const details = leiden.detailed(graph, { resolution: 1.0, randomWalk: true });
    console.log(`[semantic-clusters] modularity=${details.modularity.toFixed(3)}`);

    const communityMembers = new Map<number, { ref_id: string; name: string; file: string }[]>();
    for (const [ref_id, commNum] of Object.entries(details.communities as Record<string, number>)) {
      if (!communityMembers.has(commNum)) communityMembers.set(commNum, []);
      const attrs = graph.getNodeAttributes(ref_id);
      communityMembers.get(commNum)!.push({ ref_id, name: attrs.name, file: attrs.file });
    }

    const finalClusters: { clusterId: string; label: string; memberIds: string[] }[] = [];
    for (const [commNum, members] of communityMembers) {
      if (members.length < MIN_COMMUNITY_SIZE) continue;
      const label = generateSemanticLabel(members, commNum);
      finalClusters.push({
        clusterId: `semantic_${commNum}`,
        label,
        memberIds: members.map(m => m.ref_id),
      });
    }

    await db.clear_semantic_clusters();

    await db.bulk_upsert_clusters(
      finalClusters.map(({ clusterId, label, memberIds }) => ({
        cluster_id: clusterId,
        label,
        cohesion: 0,
        symbol_count: memberIds.length,
      }))
    );

    const memberEdges = finalClusters.flatMap(({ clusterId, memberIds }) =>
      memberIds.map((refId) => ({ ref_id: refId, cluster_id: clusterId }))
    );
    await db.bulk_create_member_of(memberEdges);

    console.log(`[semantic-clusters] done: ${finalClusters.length} clusters`);
    return { clusterCount: finalClusters.length, modularity: details.modularity, nodesProcessed: graph.order };
  } finally {
    await db.drop_semantic_graph(GDS_GRAPH_NAME);
  }
}
