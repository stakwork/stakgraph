import { Router } from "express";
import { list_sessions, get_session } from "./sessions.js";

export function benchmarkRouter(): Router {
  const router = Router();

  // Sessions
  router.get("/sessions", list_sessions);
  router.get("/sessions/:id", get_session);

  return router;
}
