import { Router } from "express";
import { list_sessions, list_session_facets, get_session, get_session_turns, session_stats, add_annotation } from "./sessions.js";

export function benchmarkRouter(): Router {
  const router = Router();

  // Sessions
  router.get("/sessions", list_sessions);
  router.get("/sessions/facets", list_session_facets);
  router.get("/sessions/stats", session_stats);
  router.get("/sessions/:id", get_session);
  router.get("/sessions/:id/turns", get_session_turns);
  router.post("/sessions/:id/annotations", add_annotation);

  return router;
}
