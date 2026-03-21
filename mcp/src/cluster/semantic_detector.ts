import GraphModule from 'graphology';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { db } from '../graph/neo4j.js';
import { ClusterDetectionResult } from './detector.js';
import { generateLabel, detectLang } from './lang/index.js';

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
const STRUCTURAL_BOOST = 0.15;
const DOMAIN_RESOLUTION = 0.1;
const MIN_DOMAIN_SIZE = 2;

export async function runSemanticClusterDetection(): Promise<ClusterDetectionResult> {
  try {
    const nodeCount = await db.project_semantic_graph(GDS_GRAPH_NAME);
    console.log(`[semantic-clusters] projected ${nodeCount} nodes`);

    if (nodeCount === 0) {
      return { clusterCount: 0, modularity: 0, nodesProcessed: 0 };
    }

    const pairs = await db.stream_semantic_knn(
      GDS_GRAPH_NAME,
      KNN_TOP_K,
      KNN_SAMPLE_RATE,
      KNN_SIMILARITY_CUTOFF,
    );
    console.log(
      `[semantic-clusters] KNN returned ${pairs.length} similarity pairs`,
    );

    // Build graphology graph from KNN pairs for vendored Leiden
    const graph: GraphInstance = new Graph({
      type: "undirected",
      allowSelfLoops: false,
    });
    const nodeAttrs = new Map<string, { name: string; file: string }>();

    for (const p of pairs) {
      if (p.source && !nodeAttrs.has(p.source)) {
        nodeAttrs.set(p.source, { name: p.sourceName, file: p.sourceFile });
      }
      if (p.target && !nodeAttrs.has(p.target)) {
        nodeAttrs.set(p.target, { name: p.targetName, file: p.targetFile });
      }
    }
    for (const [ref_id, attrs] of nodeAttrs) {
      graph.addNode(ref_id, attrs);
    }
    for (const p of pairs) {
      if (
        p.source &&
        p.target &&
        graph.hasNode(p.source) &&
        graph.hasNode(p.target) &&
        p.source !== p.target
      ) {
        if (!graph.hasEdge(p.source, p.target)) {
          graph.addEdge(p.source, p.target, { weight: p.similarity });
        }
      }
    }

    console.log(
      `[semantic-clusters] graph: ${graph.order} nodes, ${graph.size} edges`,
    );

    const structuralEdges = await db.get_semantic_structural_edges();
    let boosted = 0;
    for (const e of structuralEdges) {
      if (!e.source || !e.target || e.source === e.target) continue;
      if (graph.hasEdge(e.source, e.target)) {
        const w = graph.getEdgeAttribute(
          graph.edge(e.source, e.target),
          "weight",
        );
        graph.setEdgeAttribute(
          graph.edge(e.source, e.target),
          "weight",
          w + STRUCTURAL_BOOST,
        );
        boosted++;
      } else if (graph.hasNode(e.source) && graph.hasNode(e.target)) {
        graph.addEdge(e.source, e.target, { weight: STRUCTURAL_BOOST });
        boosted++;
      }
    }
    console.log(
      `[semantic-clusters] structural boost: ${boosted} edges boosted/added (factor=${STRUCTURAL_BOOST})`,
    );

    const details = leiden.detailed(graph, {
      resolution: 1.0,
      randomWalk: true,
    });
    console.log(
      `[semantic-clusters] modularity=${details.modularity.toFixed(3)}, levels=${details.level}`,
    );

    const communityMembers = new Map<
      number,
      { ref_id: string; name: string; file: string }[]
    >();
    for (const [ref_id, commNum] of Object.entries(
      details.communities as Record<string, number>,
    )) {
      if (!communityMembers.has(commNum)) communityMembers.set(commNum, []);
      const attrs = graph.getNodeAttributes(ref_id);
      communityMembers
        .get(commNum)!
        .push({ ref_id, name: attrs.name, file: attrs.file });
    }

    const allFiles = [...communityMembers.values()].flat().map((m) => m.file);
    const lang = detectLang(allFiles);
    const usedLabels = new Set<string>();
    const finalClusters: {
      clusterId: string;
      label: string;
      memberIds: string[];
    }[] = [];
    for (const [commNum, members] of communityMembers) {
      if (members.length < MIN_COMMUNITY_SIZE) continue;
      const files = members.map((m) => m.file);
      const label = generateLabel(
        files,
        commNum,
        usedLabels,
        lang,
        "SemanticCluster",
      );
      finalClusters.push({
        clusterId: `semantic_${commNum}`,
        label,
        memberIds: members.map((m) => m.ref_id),
      });
    }

    // --- Domain clusters (SemanticDomain) via second Leiden pass at lower resolution ---
    // Run Leiden again at DOMAIN_RESOLUTION to get larger, coarser groups.
    const coarseDetails = leiden.detailed(graph, {
      resolution: DOMAIN_RESOLUTION,
      randomWalk: true,
    });
    const fineComm = details.communities as Record<string, number>;
    const domainComm = coarseDetails.communities as Record<string, number>;
    const domainCount = new Set(Object.values(domainComm)).size;
    console.log(
      `[semantic-clusters] domain pass: ${domainCount} domain communities (resolution=${DOMAIN_RESOLUTION})`,
    );

    // Map each fine community ID → domain community ID (via nodes)
    const fineToDomain = new Map<number, number>();
    for (const [ref_id, fineId] of Object.entries(fineComm)) {
      if (!fineToDomain.has(fineId) && domainComm[ref_id] !== undefined) {
        fineToDomain.set(fineId, domainComm[ref_id]);
      }
    }

    // Build domain membership from fine clusters that passed the size filter
    const domainToFineClusters = new Map<number, string[]>();
    for (const { clusterId } of finalClusters) {
      const commNum = parseInt(clusterId.replace("semantic_", ""), 10);
      const domainId = fineToDomain.get(commNum);
      if (domainId === undefined) continue;
      if (!domainToFineClusters.has(domainId))
        domainToFineClusters.set(domainId, []);
      domainToFineClusters.get(domainId)!.push(clusterId);
    }

    const domainUsedLabels = new Set<string>();
    const domainClusters: {
      clusterId: string;
      label: string;
      childClusterIds: string[];
    }[] = [];
    for (const [domainCommNum, childClusterIds] of domainToFineClusters) {
      if (childClusterIds.length < MIN_DOMAIN_SIZE) continue;
      // Collect all files from member fine clusters
      const domainFiles = childClusterIds.flatMap((cid) => {
        const fc = finalClusters.find((f) => f.clusterId === cid);
        return fc
          ? fc.memberIds.flatMap((rid) => {
              const n = graph.getNodeAttributes(rid);
              return n.file ? [n.file as string] : [];
            })
          : [];
      });
      const domainLabel = generateLabel(
        domainFiles,
        domainCommNum,
        domainUsedLabels,
        lang,
        "SemanticDomain",
      );
      domainClusters.push({
        clusterId: `domain_${domainCommNum}`,
        label: domainLabel,
        childClusterIds,
      });
    }
    console.log(
      `[semantic-clusters] ${domainClusters.length} domain clusters from dendrogram`,
    );

    // --- Persist: clear old, write fine clusters, write domains, link hierarchy ---
    await db.clear_semantic_domains();
    await db.clear_semantic_clusters();

    await db.bulk_upsert_clusters(
      finalClusters.map(({ clusterId, label, memberIds }) => ({
        cluster_id: clusterId,
        label,
        cohesion: 0,
        symbol_count: memberIds.length,
      })),
    );

    const memberEdges = finalClusters.flatMap(({ clusterId, memberIds }) =>
      memberIds.map((refId) => ({ ref_id: refId, cluster_id: clusterId })),
    );
    await db.bulk_create_member_of(memberEdges);

    await db.bulk_upsert_clusters(
      domainClusters.map(({ clusterId, label, childClusterIds }) => ({
        cluster_id: clusterId,
        label,
        cohesion: 0,
        symbol_count: childClusterIds.length,
      })),
    );

    const parentEdges = domainClusters.flatMap(
      ({ clusterId, childClusterIds }) =>
        childClusterIds.map((childId) => ({
          parent_id: clusterId,
          child_id: childId,
        })),
    );
    await db.bulk_create_cluster_parent_of(parentEdges);

    console.log(
      `[semantic-clusters] done: ${finalClusters.length} clusters, ${domainClusters.length} domains`,
    );
    return {
      clusterCount: finalClusters.length,
      modularity: details.modularity,
      nodesProcessed: graph.order,
      domainCount: domainClusters.length,
    };
  } finally {
    await db.drop_semantic_graph(GDS_GRAPH_NAME);
  }
}
