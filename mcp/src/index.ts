// Global console patch - add ISO 8601 timestamps to all console output
const _ts = () => new Date().toISOString();
const _log = console.log.bind(console);
const _warn = console.warn.bind(console);
const _error = console.error.bind(console);
const _debug = console.debug.bind(console);
const _info = console.info.bind(console);
const _trace = console.trace.bind(console);
console.log   = (...a: unknown[]) => _log(_ts(), ...a);
console.warn  = (...a: unknown[]) => _warn(_ts(), ...a);
console.error = (...a: unknown[]) => _error(_ts(), ...a);
console.debug = (...a: unknown[]) => _debug(_ts(), ...a);
console.info  = (...a: unknown[]) => _info(_ts(), ...a);
console.trace = (...a: unknown[]) => _trace(_ts(), ...a);

import express, { Request, Response } from "express";
import { graph_mcp_routes } from "./tools/server.js";
import { graph_sse_routes } from "./tools/sse.js";
import fileUpload from "express-fileupload";
import * as r from "./graph/routes.js";
import * as l from "./graph/learnings.js";
import * as uploads from "./graph/uploads.js";
import * as gitree from "./gitree/routes.js";
import path from "path";
import { fileURLToPath } from "url";
import cors from "cors";
import { App as SageApp } from "./sage/src/app.js";
import dotenv from "dotenv";
import { learn_docs_agent, get_docs, update_docs } from "./repo/docs.js";
import { document_workflow, document_workflows } from "./repo/workflows.js";
import { cacheMiddleware, cacheInfo, clearCache } from "./graph/cache.js";
import { evalRoutes } from "./eval/route.js";
import { test_routes } from "./eval/tests.js";
import * as rr from "./repo/index.js";
import * as cluster from "./cluster/index.js";
import { getBusy, busyMiddleware } from "./busy.js";
import { mcp_routes } from "./handler/index.js";
import { logs_agent } from "./log/index.js";
import { pruneExpiredSessions } from "./repo/session.js";
import { getBus, verifyEventsToken, pipeToSSE } from "./repo/events.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function swagger(_: Request, res: Response) {
  res.sendFile(path.join(__dirname, "../docs/redoc-static.html"));
}

const app = express();
app.use(cors());

// SSE routes must come before body parsing middleware to preserve raw streams
graph_sse_routes(app);

// Real-time agent step events via SSE
// Auth: JWT query param (?token=xxx) scoped to the request_id, OR x-api-token header
app.get("/events/:request_id", (req: Request, res: Response) => {
  const request_id = req.params.request_id as string;
  const jwtToken = req.query.token as string | undefined;
  const apiToken = process.env.API_TOKEN;
  const headerToken = req.header("x-api-token");

  // Auth check: require either a valid JWT or a matching x-api-token
  let authorized = false;

  if (jwtToken) {
    try {
      const payload = verifyEventsToken(jwtToken);
      if (payload.request_id === request_id) authorized = true;
    } catch (_) {
      // invalid/expired JWT
    }
  }

  if (!authorized && apiToken && headerToken === apiToken) {
    authorized = true;
  }

  // If no API_TOKEN is set (dev mode), allow unauthenticated access
  if (!apiToken) authorized = true;

  if (!authorized) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  const bus = getBus(request_id);
  if (!bus) {
    res.status(404).json({ error: "No active event stream for this request_id" });
    return;
  }

  pipeToSSE(bus, res);
});

app.use(express.json({ limit: "50mb" }));

graph_mcp_routes(app);

mcp_routes(app);

app.use(express.urlencoded({ extended: true }));
app.use(fileUpload());

try {
  new SageApp(app);
} catch (e) {
  console.log("===> skipping sage setup");
}

app.get("/", swagger);
app.use("/textarea", express.static(path.join(__dirname, "../textarea")));
app.use("/app", express.static(path.join(__dirname, "../app")));
app.use("/demo", express.static(path.join(__dirname, "../app/vendor")));
app.get("/schema", r.schema);
app.get("/ontology", r.schema);

evalRoutes(app);

test_routes(app);

// Learn route needs to handle its own authentication
app.get("/learn", r.learn);

app.get("/busy", (req: Request, res: Response) => {
  res.json({ busy: getBusy() });
});

app.get("/gitsee/events/:owner/:repo", r.gitseeEvents);

app.use(r.authMiddleware);
app.use(r.logEndpoint);
app.get("/nodes", r.get_nodes);
app.post("/nodes", r.post_nodes);
app.get("/edges", r.get_edges);
app.get("/graph", r.get_graph);
app.get("/search", r.search);
app.get("/map", r.get_map);
app.get("/repo_map", cacheMiddleware(), r.get_repo_map);
app.get("/code", r.get_code);
app.get("/shortest_path", r.get_shortest_path);
app.post("/upload", uploads.upload_files);
app.get("/status/:requestId", uploads.check_status);
app.get("/update_token_counts", uploads.update_token_counts);
app.get("/rules_files", r.get_rules_files);
app.get("/services", r.get_services);
app.get("/explore", busyMiddleware, r.explore);
app.get("/understand", busyMiddleware, r.understand);
app.post("/seed_understanding", busyMiddleware, r.seed_understanding);
app.get("/ask", r.ask);
app.get("/learnings_og", r.get_learnings_og);
app.post("/learnings", l.post_learnings);
app.get("/learnings/all", l.get_all_learnings);
app.post("/relevant-learnings", l.post_relevant_learnings);
app.get("/subgraph", r.fetch_node_with_related);
app.get("/workflow", r.fetch_workflow_published_version);
app.post("/hint_siblings", r.generate_siblings);
app.post("/seed_stories", r.seed_stories);
app.get("/services_agent", rr.services_agent);
app.get("/mocks", rr.mocks_agent);
app.get("/mocks/inventory", r.mocks_inventory);
app.get("/agent", busyMiddleware, r.gitsee_agent);
app.post("/gitsee", r.gitsee);
app.get("/progress", r.get_script_progress);
app.get("/leaks", rr.get_leaks);
app.post("/repo/agent", rr.repo_agent);
app.get("/repo/agent/tools", rr.get_agent_tools);
app.get("/repo/agent/session", rr.get_agent_session);
app.get("/repo/agent/validate_session", rr.validate_agent_session);
app.get("/repo/agent/file", rr.get_agent_file);
app.post("/repo/describe", rr.describe_nodes_agent);
app.get("/reattach", r.reconnect_orphaned_hints);
app.post("/pull_request", r.create_pull_request);
app.post("/learn_docs", learn_docs_agent);
app.get("/docs", get_docs);
app.put("/docs", update_docs);
app.post("/logs/agent", logs_agent);

// Gitree routes
app.post("/gitree/process", gitree.gitree_process);
app.get("/gitree/features", gitree.gitree_list_features);
app.get("/gitree/features/:id", gitree.gitree_get_feature);
app.put("/gitree/features/:id/documentation", gitree.gitree_update_feature_documentation);
app.delete("/gitree/features/:id", gitree.gitree_delete_feature);
app.get("/gitree/features/:id/files", gitree.gitree_get_feature_files);
app.get("/gitree/prs/:number", gitree.gitree_get_pr);
app.get("/gitree/commits/:sha", gitree.gitree_get_commit);
app.get("/gitree/stats", gitree.gitree_stats);
app.get("/gitree/all-features-graph", gitree.gitree_all_features_graph);
app.post("/gitree/summarize/:id", gitree.gitree_summarize_feature);
app.post("/gitree/summarize-all", gitree.gitree_summarize_all);
app.post("/gitree/link-files", gitree.gitree_link_files);
app.post("/gitree/relevant-features", gitree.gitree_relevant_features);
app.post("/gitree/create-feature", gitree.gitree_create_feature);
app.post("/gitree/analyze-clues", gitree.gitree_analyze_clues);
app.post("/gitree/analyze-changes", gitree.gitree_analyze_changes);
app.post("/gitree/link-clues", gitree.gitree_link_clues);
app.get("/gitree/clues", gitree.gitree_list_clues);
app.get("/gitree/clues/:id", gitree.gitree_get_clue);
app.delete("/gitree/clues/:id", gitree.gitree_delete_clue);
app.post("/gitree/search-clues", gitree.gitree_search_clues);
app.post("/gitree/provenance", gitree.gitree_provenance);

app.get("/clusters", cluster.list_clusters);
app.post("/clusters/detect", cluster.detect_clusters);
app.post("/clusters/semantic/detect", cluster.detect_semantic_clusters);
app.delete("/clusters", cluster.clear_clusters_route);
app.get("/clusters/:cluster_id/members", cluster.get_cluster_members_route);

app.post("/document_workflow", document_workflow);
app.post("/document_workflows", document_workflows);

app.get("/_cache/info", cacheInfo);
app.post("/_cache/clear", (_req: Request, res: Response): void => {
  clearCache();
  res.json({ message: "Cache cleared" });
});

const port = parseInt(process.env.PORT || '3355', 10);
const host = process.env.HOST || '0.0.0.0';
app.listen(port, host, () => {
  console.log(`Server started at http://${host}:${port}`);

  // Prune expired sessions on startup, then every 6 hours
  pruneExpiredSessions();
  setInterval(pruneExpiredSessions, 6 * 60 * 60 * 1000);
});

//
