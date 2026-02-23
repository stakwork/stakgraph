import { Request, Response } from "express";
import {
  ContainerConfig,
} from "../types.js";
import {
  detectLanguagesAndPkgFiles,
  cloneRepoToTmp,
  extractEnvVarsFromRepo,
  findDockerComposeFiles,
  normalizeRepoParam,
} from "../utils.js";
import fs from "fs/promises";
import * as G from "../graph.js";
import { parseServiceFile, extractContainersFromCompose } from "../service.js";
import * as path from "path";

export async function get_services(req: Request, res: Response) {
  try {
    if (req.query.clone === "true" && req.query.repo_url) {
      const repoUrl = req.query.repo_url as string;
      const username = req.query.username as string | undefined;
      const pat = req.query.pat as string | undefined;
      const commit = req.query.commit as string | undefined;

      const repoDir = await cloneRepoToTmp(repoUrl, username, pat, commit);
      const detected = await detectLanguagesAndPkgFiles(repoDir);

      const envVarsByFile = await extractEnvVarsFromRepo(repoDir);

      const services = [];
      for (const { language, pkgFile } of detected) {
        const body = await fs.readFile(pkgFile, "utf8");
        const service = parseServiceFile(pkgFile, body, language);

        const serviceDir = path.dirname(pkgFile);
        const envVars = new Set<string>();
        for (const [file, vars] of Object.entries(envVarsByFile)) {
          if (file.startsWith(serviceDir)) {
            vars.forEach((v) => envVars.add(v));
          }
        }

        service.env = {};
        envVars.forEach((v) => (service.env[v] = process.env[v] || ""));

        const { pkgFile: _, ...cleanService } = service;
        services.push(cleanService);
      }
      const composeFiles = await findDockerComposeFiles(repoDir);
      let containers: ContainerConfig[] = [];
      for (const composeFile of composeFiles) {
        const found = await extractContainersFromCompose(composeFile);
        containers = containers.concat(found);
      }
      res.json({ services, containers });
      return;
    } else {
      const { services, containers } = await G.get_services();
      res.json({ services, containers });
    }
  } catch (error) {
    console.error("Error getting services config:", error);
    res
      .status(500)
      .json({ error: "Failed to generate services configuration" });
  }
}

export async function mocks_inventory(req: Request, res: Response) {
  try {
    const search = req.query.search as string | undefined;
    const repo = normalizeRepoParam(req.query.repo as string | undefined);
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const result = await G.get_mocks_inventory(search, limit, offset, repo);
    res.json(result);
  } catch (error) {
    console.error("Error getting mocks inventory:", error);
    res.status(500).json({ error: "Failed to get mocks inventory" });
  }
}
