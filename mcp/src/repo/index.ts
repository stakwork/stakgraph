import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { ToolsConfig, getDefaultToolDescriptions } from "./tools.js";
import { Request, Response } from "express";
import { gitleaksDetect, gitleaksProtect } from "./gitleaks.js";

export async function repo_agent(req: Request, res: Response) {
  console.log("===> repo_agent", req.body);
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

    const repoDir = await cloneOrUpdateRepo(repoUrl, username, pat, commit);

    console.log(`===> POST /repo/agent ${repoDir}`);

    const result = await get_context(prompt, repoDir, pat, toolsConfig);

    // console.log("===> final_answer", result.final);
    res.json({ success: true, final_answer: result.final, usage: result.usage });
  } catch (e) {
    console.error("Error in repo_agent", e);
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
