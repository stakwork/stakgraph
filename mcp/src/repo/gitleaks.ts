import { execSync } from "child_process";
import { Request, Response } from "express";
import { cloneOrUpdateRepo } from "./clone.js";

export interface GitLeakResult {
  Author: string;
  Commit: string;
  Date: string;
  Description: string;
  Email: string;
  EndColumn: number;
  EndLine: number;
  Entropy: number;
  File: string;
  Fingerprint: string;
  Link: string;
  Match: string;
  Message: string;
  RuleID: string;
  Secret: string;
  StartColumn: number;
  StartLine: number;
  SymlinkFile: string;
  Tags: string[];
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

const CMD_END =
  "--no-banner --no-color --log-level=fatal --report-format=json --report-path=-";

function filterResults(
  results: GitLeakResult[],
  ignore: string[]
): GitLeakResult[] {
  if (ignore.length === 0) {
    return results;
  }

  return results.filter((result) => {
    // Check if the file path contains any of the ignored directories
    return !ignore.some((dir) => {
      const pathParts = result.File.split("/");
      return pathParts.includes(dir);
    });
  });
}

export function gitleaksDetect(
  repoDir: string,
  ignore: string[] = []
): GitLeakResult[] {
  const cmd = "gitleaks detect " + CMD_END;
  try {
    const result = execSync(cmd, {
      encoding: "utf-8",
      cwd: repoDir,
    });
    const parsed = JSON.parse(result) as GitLeakResult[];
    return filterResults(parsed, ignore);
  } catch (error: any) {
    // If gitleaks found secrets, it exits with code 1 but still outputs valid JSON
    if (error.status === 1 && error.stdout) {
      const parsed = JSON.parse(error.stdout) as GitLeakResult[];
      return filterResults(parsed, ignore);
    }
    // Re-throw for other types of errors
    throw error;
  }
}

export function gitleaksProtect(
  repoDir: string,
  ignore: string[] = []
): GitLeakResult[] {
  const cmd = "gitleaks protect " + CMD_END;
  try {
    const result = execSync(cmd, {
      encoding: "utf-8",
      cwd: repoDir,
    });
    const parsed = JSON.parse(result) as GitLeakResult[];
    return filterResults(parsed, ignore);
  } catch (error: any) {
    // If gitleaks found secrets, it exits with code 1 but still outputs valid JSON
    if (error.status === 1 && error.stdout) {
      const parsed = JSON.parse(error.stdout) as GitLeakResult[];
      return filterResults(parsed, ignore);
    }
    // Re-throw for other types of errors
    throw error;
  }
}
