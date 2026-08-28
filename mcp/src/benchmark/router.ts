import { Router } from "express";
import { list_sessions, list_session_facets, get_session, get_session_turns, session_stats, add_annotation } from "./sessions.js";
import {
  create_session,
  append_turns,
  end_session,
  link_session_concepts,
} from "./ingest.js";
import { authMiddleware } from "../graph/routes.js";

export function benchmarkRouter(): Router {
  const router = Router();

  // Sessions
  router.get("/sessions", list_sessions);
  router.get("/sessions/facets", list_session_facets);
  router.get("/sessions/stats", session_stats);
  router.get("/sessions/:id", get_session);
  router.get("/sessions/:id/turns", get_session_turns);
  router.post("/sessions/:id/annotations", authMiddleware, add_annotation);

  // Ingest — out-of-process agents recording live Turn chains (see ingest.ts
  // and docs/session-ingest.md). Auth-gated, unlike the read endpoints these
  // sit beside: these write to the graph. authMiddleware is a no-op when
  // API_TOKEN is unset (dev), so local callers are unaffected.
  router.post("/sessions", authMiddleware, create_session);
  router.post("/sessions/:id/turns", authMiddleware, append_turns);
  router.post("/sessions/:id/end", authMiddleware, end_session);
  router.post("/sessions/:id/concepts", authMiddleware, link_session_concepts);

  return router;
}
