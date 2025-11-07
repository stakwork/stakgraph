import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { ToolsConfig, getDefaultToolDescriptions } from "./tools.js";
import { Request, Response } from "express";
import { gitleaksDetect, gitleaksProtect } from "./gitleaks.js";
import * as asyncReqs from "../graph/reqs.js";

export async function repo_agent(req: Request, res: Response) {
  // curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"
  // curl "http://localhost:3355/progress?request_id=123"
  console.log("===> repo_agent", req.body, req.body.prompt);
  const request_id = asyncReqs.startReq();
  try {
    const repoUrl = req.body.repo_url as string;
    const username = req.body.username as string | undefined;
    const pat = req.body.pat as string | undefined;
    const commit = req.body.commit as string | undefined;
    const branch = req.body.branch as string | undefined;
    const prompt = req.body.prompt as string | undefined;
    const toolsConfig = req.body.toolsConfig as ToolsConfig | undefined;
    if (!prompt) {
      res.status(400).json({ error: "Missing prompt" });
      return;
    }

    const { setBusy } = await import("../busy.js");
    setBusy(true);
    console.log("[repo_agent] Set busy=true before starting work");

    cloneOrUpdateRepo(repoUrl, username, pat, commit)
      .then((repoDir) => {
        console.log(`===> POST /repo/agent ${repoDir}`);
        return get_context(prompt, repoDir, pat, toolsConfig);
      })
      .then((result) => {
        asyncReqs.finishReq(request_id, {
          success: true,
          final_answer: result.final,
          usage: result.usage,
        });
        setBusy(false);
        console.log("[repo_agent] Background work completed, set busy=false");
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
        setBusy(false);
        console.log("[repo_agent] Background work failed, set busy=false");
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    console.error("Error in repo_agent", error);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function get_agent_tools(req: Request, res: Response) {
  console.log("===> GET /repo/agent/tools");
  try {
    const toolsConfig = getDefaultToolDescriptions();
    res.json({ success: true, toolsConfig });
  } catch (e) {
    console.error("Error in get_agent_tools", e);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function get_leaks(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  const username = req.query.username as string | undefined;
  const pat = req.query.pat as string | undefined;
  const commit = req.query.commit as string | undefined;
  const ignore = req.query.ignore as string | undefined;

  const repoDir = await cloneOrUpdateRepo(repoUrl, username, pat, commit);

  console.log(`===> GET /leaks ${repoDir}`);
  try {
    const ignoreList = ignore?.split(",").map((dir) => dir.trim()) || [];
    const detect = gitleaksDetect(repoDir, ignoreList);
    const protect = gitleaksProtect(repoDir, ignoreList);
    res.json({ success: true, detect, protect });
  } catch (e) {
    console.error("Error running gitleaks:", e);
    res.status(500).json({ error: "Error running gitleaks" });
  }
}
