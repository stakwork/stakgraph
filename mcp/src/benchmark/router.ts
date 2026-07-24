import { Router } from "express";
import { list_sessions, get_session, session_stats, add_annotation, search_sessions } from "./sessions.js";

export function benchmarkRouter(): Router {
  const router = Router();

  // Sessions
  router.get("/sessions", list_sessions);
  router.get("/sessions/search", search_sessions);
  router.get("/sessions/stats", session_stats);
  router.get("/sessions/:id", get_session);
  router.post("/sessions/:id/annotations", add_annotation);

  return router;
}
