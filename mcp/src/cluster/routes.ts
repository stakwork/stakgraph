import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { runClusterDetection } from "./detector.js";
import { runSemanticClusterDetection } from "./semantic_detector.js";

export async function list_clusters(_req: Request, res: Response) {
  try {
    const clusters = await db.get_all_clusters();
    res.json({ clusters, total: clusters.length });
  } catch (e: any) {
    console.error("[clusters] list error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function detect_clusters(_req: Request, res: Response) {
  try {
    console.log("[clusters] Starting cluster detection...");
    const result = await runClusterDetection();
    console.log(`[clusters] Done: ${result.clusterCount} clusters, ${result.nodesProcessed} nodes`);
    res.json(result);
  } catch (e: any) {
    console.error("[clusters] detect error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function clear_clusters_route(_req: Request, res: Response) {
  try {
    await db.clear_clusters();
    res.json({ success: true });
  } catch (e: any) {
    console.error("[clusters] clear error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function detect_semantic_clusters(_req: Request, res: Response) {
  try {
    console.log("[semantic-clusters] Starting semantic cluster detection...");
    const result = await runSemanticClusterDetection();
    console.log(`[semantic-clusters] Done: ${result.clusterCount} clusters, ${result.nodesProcessed} nodes`);
    res.json(result);
  } catch (e: any) {
    console.error("[semantic-clusters] detect error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function get_cluster_members_route(req: Request, res: Response) {
  try {
    const cluster_id = req.params.cluster_id;
    const limit = parseInt(req.query.limit as string) || 50;
    const members = await db.get_cluster_members(cluster_id, limit);
    res.json({ cluster_id, members, total_members: members.length });
  } catch (e: any) {
    console.error("[clusters] members error:", e);
    res.status(500).json({ error: e.message });
  }
}
