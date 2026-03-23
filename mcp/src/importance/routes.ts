import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { runImportanceScoring } from "./detector.js";
import { ImportanceTag, IMPORTANCE_TAGS } from "./types.js";

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
