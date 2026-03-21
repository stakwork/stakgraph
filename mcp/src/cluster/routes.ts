import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { runClusterDetection } from "./detector.js";
import { runSemanticClusterDetection } from "./semantic_detector.js";
import { runImportanceScoring } from "./importance_detector.js";
import { ImportanceTag, IMPORTANCE_TAGS } from "./types.js";

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

export async function get_semantic_hierarchy(_req: Request, res: Response) {
  try {
    const hierarchy = await db.get_semantic_hierarchy();
    res.json({ hierarchy, total_domains: hierarchy.length });
  } catch (e: any) {
    console.error("[semantic-clusters] hierarchy error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function get_semantic_domains(_req: Request, res: Response) {
  try {
    const domains = await db.get_semantic_domains();
    res.json({ domains, total: domains.length });
  } catch (e: any) {
    console.error("[semantic-clusters] domains error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function get_domain_cluster_members_route(req: Request, res: Response) {
  try {
    const cluster_id = req.params.domain_id;
    const members = await db.get_domain_cluster_members(cluster_id);
    res.json({ cluster_id, members, total: members.length });
  } catch (e: any) {
    console.error("[semantic-clusters] domain members error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function score_importance(_req: Request, res: Response) {
  try {
    console.log("[importance] Starting importance scoring...");
    const result = await runImportanceScoring();
    console.log(`[importance] Done: ${result.nodesScored} nodes scored`);
    res.json(result);
  } catch (e: any) {
    console.error("[importance] score error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function get_top_importance(req: Request, res: Response) {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const nodes = await db.get_top_nodes_by_importance(limit);
    res.json({ nodes, total: nodes.length });
  } catch (e: any) {
    console.error("[importance] top error:", e);
    res.status(500).json({ error: e.message });
  }
}

export async function get_importance_tag(req: Request, res: Response) {
  try {
    const tag = req.query.tag as string;
    if (!tag || !IMPORTANCE_TAGS.includes(tag as ImportanceTag)) {
      res.status(400).json({
        error: `tag query param required, must be one of: ${IMPORTANCE_TAGS.join(", ")}`,
      });
      return;
    }
    const limit = parseInt(req.query.limit as string) || 50;
    const nodes = await db.get_nodes_by_importance_tag(tag as ImportanceTag, limit);
    res.json({ tag, nodes, total: nodes.length });
  } catch (e: any) {
    console.error("[importance] tag error:", e);
    res.status(500).json({ error: e.message });
  }
}
