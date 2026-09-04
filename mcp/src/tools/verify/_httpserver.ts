import express from "express";
import { graph_mcp_routes } from "../server.js";

const PORT = parseInt(process.env.MCP_HTTP_PORT || "3458", 10);
const app = express();
app.use(express.json({ limit: "16mb" }));
app.get("/health", (_req, res) => res.json({ ok: true }));
graph_mcp_routes(app);
app.listen(PORT, () => console.error(`[verify-http-mcp] listening on http://localhost:${PORT}/graph_mcp`));
