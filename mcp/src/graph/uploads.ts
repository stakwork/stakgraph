import fs from "fs";
import { v4 as uuidv4 } from "uuid";
import { db } from "./neo4j.js";
import type { Request, Response } from "express";
import { isTrue } from "./utils.js";

const UPLOAD_PATH = "./uploads/";

const jobStatus = new Map<
  string,
  {
    status: "pending" | "processing" | "completed" | "failed";
    result?: any;
    error?: string;
  }
>();

export async function upload_files(req: Request, res: Response) {
  if (!req.files || Object.keys(req.files).length === 0) {
    res.status(400).send("No files were uploaded.");
    return;
  }
  const requestId = uuidv4();
  const file1 = req.files.nodes;
  const file2 = req.files.edges;
  try {
    // Check if files are arrays or single files
    const nodesFiles = Array.isArray(file1) ? file1 : [file1];
    const edgesFiles = Array.isArray(file2) ? file2 : [file2];

    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(UPLOAD_PATH)) {
      fs.mkdirSync(UPLOAD_PATH, { recursive: true });
    }

    const node_file = UPLOAD_PATH + nodesFiles[0].name;
    await nodesFiles[0].mv(node_file);

    const edge_file = UPLOAD_PATH + edgesFiles[0].name;
    await edgesFiles[0].mv(edge_file);

    jobStatus.set(requestId, { status: "pending" });
    processFiles(requestId, node_file, edge_file).catch((error) => {
      jobStatus.set(requestId, {
        status: "failed",
        error: error.message,
      });
    });

    res.json({
      status: "success",
      message: "Files uploaded successfully",
      requestId: requestId,
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("Error uploading files");
  }
}

export async function check_status(req: Request, res: Response) {
  const { requestId } = req.params;
  const status = jobStatus.get(requestId);
  if (!status) {
    res.status(404).json({ error: "Request ID not found" });
    return;
  }
  res.json(status);
}

async function processFiles(requestId: string, node_file: any, edge_file: any) {
  jobStatus.set(requestId, { status: "processing" });
  await db.build_graph_from_files(node_file, edge_file);
  console.log("Graph built and code embedded");
  jobStatus.set(requestId, { status: "completed" });
}

export async function embed_code(req: Request, res: Response) {
  const do_files = isTrue(req.query.files as string);
  await db.embed_data_bank_bodies(do_files);
  res.json({ status: "completed" });
}
