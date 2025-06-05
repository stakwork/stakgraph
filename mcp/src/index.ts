import express, { Request, Response } from "express";
import { mcp_routes } from "./tools/index.js";
import fileUpload from "express-fileupload";
import * as r from "./graph/routes.js";
import * as uploads from "./graph/uploads.js";
import path from "path";
import { fileURLToPath } from "url";
import cors from "cors";
import { App as SageApp } from "./sage/src/app.js";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function swagger(_: Request, res: Response) {
  res.sendFile(path.join(__dirname, "../docs/redoc-static.html"));
}

const app = express();
app.use(cors());

// MCP routes must come before body parsing middleware to preserve raw streams
mcp_routes(app);

app.use(express.json());
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

app.use(r.authMiddleware);
app.use(r.logEndpoint);
app.get("/nodes", r.get_nodes);
app.get("/search", r.search);
app.get("/map", r.get_map);
app.get("/repo_map", r.get_repo_map);
app.get("/code", r.get_code);
app.get("/shortest_path", r.get_shortest_path);
app.post("/upload", uploads.upload_files);
app.get("/status/:requestId", uploads.check_status);
app.get("/embed_code", uploads.embed_code);
app.get("/update_token_counts", uploads.update_token_counts);

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started at http://localhost:${port}`);
});
