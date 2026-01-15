import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { generateText } from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";

export async function learn_docs_agent(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  if (!repoUrl) {
    res.status(400).json({ error: "Missing repo_url" });
    return;
  }
  const sync = req.query.sync === "true" || req.query.sync === "1";

  // Resolve AI model
  const provider = (process.env.LLM_PROVIDER || "anthropic") as Provider;
  const model = getModel(provider);

  try {
    const allRepos = await db.get_repositories();

    const processed = allRepos.find((r) => r.properties.documentation);

    if (processed) {
      console.log(`[learn_docs] Documentation already exists, skipping`);
      res.json({ message: "Documentation already exists" });
      return;
    }

    const allRulesFiles = await db.get_rules_files();

    const summaries: Record<string, string> = {};

    for (const repo of allRepos) {
      const repoName = repo.properties.name;
      const repoRoot = repo.properties.file;
      console.log(`[learn_docs] Processing ${repoName} ${repoRoot}`);

      if (repo.properties.documentation && !sync) {
        console.log(
          `[learn_docs] Documentation already exists for ${repoName}, skipping`
        );
        continue;
      }

      const repoRulesFiles = allRulesFiles.filter((f) => {
        if (repoRoot && f.properties.file.startsWith(repoRoot)) {
          return true;
        }
        if (allRepos.length === 1) {
          return true;
        }
        return false;
      });

      if (repoRulesFiles.length === 0) {
        console.log(`[learn_docs] No rules files found for ${repoName}`);
        continue;
      }

      console.log(
        `[learn_docs] Summarizing ${repoRulesFiles.length} files for ${repoName}`
      );

      const docsContent = repoRulesFiles
        .map(
          (f) => `File: ${f.properties.file}\nContent:\n${f.properties.body}`
        )
        .join("\n\n");

      const prompt = `
        Summarize the following documentation files into a high-level overview of the repository's architecture, conventions, and key patterns, etc.
        Files:
        ${docsContent}
        `;

      const result = await generateText({
        model,
        prompt,
      });

      const summary = result.text;
      await db.update_repository_documentation(
        repo.properties.ref_id!,
        summary
      );
      summaries[repoName] = summary;
      console.log(`[learn_docs] Updated documentation for ${repoName}`);
    }

    res.json({ message: "Documentation learned", summaries });
  } catch (error) {
    console.error(`[learn_docs] Error:`, error);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function get_docs(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  try {
    const repos = await db.get_repositories();
    const docs: Array<Record<string, { documentation: string }>> = [];
    for (const repo of repos) {
      if (repoUrl.includes(repo.properties.name)) {
        if (repo.properties.documentation) {
          console.log(
            `repoName: ${repo.properties.name} repoFile: ${repo.properties.file}`
          );
          docs.push({
            [repo.properties.name]: {
              documentation: repo.properties.documentation,
            },
          });
        }
      }
    }
    res.json(docs);
  } catch (error) {
    console.error("[get_docs] Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}
