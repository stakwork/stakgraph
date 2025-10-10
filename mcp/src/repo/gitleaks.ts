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
