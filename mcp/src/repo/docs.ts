import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { generateText } from "ai";
import { resolveLLMConfig } from "../aieo/src/index.js";

export async function learn_docs_agent(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  const force = req.query.force === "true";

  console.log("===> learn_docs_agent", repoUrl, "force:", force);
  const reqModel = (req.query.model || req.body?.model) as string | undefined;
  const reqApiKey = (req.query.apiKey || req.body?.apiKey) as string | undefined;
  const llm = resolveLLMConfig({ model: reqModel, apiKey: reqApiKey });
  const model = llm.model;

  try {
    const allRepos = await db.get_repositories();

    let reposToProcess = allRepos;
    if (repoUrl) {
      const repoName = repoUrl.split("/").pop()?.replace(".git", "") || "";
      reposToProcess = allRepos.filter((repo) => {
        const matchesSourceLink = repo.properties.source_link === repoUrl;
        const matchesName = repo.properties.name === repoName;
        return matchesSourceLink || matchesName;
      });

      if (reposToProcess.length === 0) {
        res.status(404).json({ error: "Repository not found" });
        return;
      }
    }

    const allRulesFiles = await db.get_rules_files();
    console.log(`[learn_docs] Total rules files in graph: ${allRulesFiles.length}`);

    const summaries: Record<string, string> = {};
    let totalInputTokens = 0, totalOutputTokens = 0;

    for (const repo of reposToProcess) {
      const repoName = repo.properties.name;
      const repoRoot = repo.properties.file;

      if (repo.properties.documentation && !force) {
        console.log(`[learn_docs] Documentation already exists for ${repoName}, skipping`);
        continue;
      }

      console.log(`[learn_docs] Processing ${repoName}, repoRoot="${repoRoot}"`);

      if (!repoRoot) {
        console.warn(`[learn_docs] No repoRoot (file property) on Repository node for ${repoName}, skipping`);
        continue;
      }

      try {
        const repoRulesFiles = allRulesFiles.filter((f) =>
          f.properties.file?.startsWith(repoRoot)
        );

        console.log(`[learn_docs] Matched ${repoRulesFiles.length}/${allRulesFiles.length} rules files for ${repoName}`);

        if (repoRulesFiles.length === 0) {
          console.log(`[learn_docs] No rules files found for ${repoName} under root "${repoRoot}"`);
          continue;
        }

        console.log(
          `[learn_docs] Summarizing ${repoRulesFiles.length} files for ${repoName}`,
        );

        const docsContent = repoRulesFiles
          .map(
            (f) => `File: ${f.properties.file}\nContent:\n${f.properties.body}`,
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
        totalInputTokens += result.usage?.inputTokens || 0;
        totalOutputTokens += result.usage?.outputTokens || 0;

        const summary = result.text;

        if (!repo.properties.ref_id) {
          throw new Error(`Repository ${repoName} missing ref_id`);
        }

        await db.update_repository_documentation(
          repo.properties.ref_id,
          summary,
        );
        summaries[repoName] = summary;
        console.log(`[learn_docs] Updated documentation for ${repoName}`);
      } catch (error) {
        console.error(`[learn_docs] Error processing ${repoName}:`, error);
      }
    }

    res.json({
      message: "Documentation learned",
      summaries,
      usage: {
        inputTokens: totalInputTokens,
        outputTokens: totalOutputTokens,
        totalTokens: totalInputTokens + totalOutputTokens,
      },
    });
  } catch (error) {
    console.error(`[learn_docs] Error:`, error);
    res.status(500).json({ error: "Internal server error" });
  }
}

/**
 * Update main docs per repo
 * PUT /docs
 * Body: { repo: string, documentation: string }
 */
export async function update_docs(req: Request, res: Response) {
  const { repo, documentation } = req.body;
  if (!repo || !documentation) {
    res.status(400).json({ error: "Missing repo or documentation" });
    return;
  }
  try {
    const allRepos = await db.get_repositories();
    const shortName = repo.split("/").pop()?.replace(".git", "") || "";
    const match = allRepos.find((r) => {
      const name = r.properties.name || "";
      return (
        r.properties.source_link === repo ||
        name === repo ||
        name.endsWith(`/${shortName}`) ||
        name === shortName
      );
    });
    if (!match) {
      res.status(404).json({ error: "Repository not found" });
      return;
    }
    if (!match.properties.ref_id) {
      res.status(500).json({ error: "Repository missing ref_id" });
      return;
    }
    await db.update_repository_documentation(
      match.properties.ref_id,
      documentation,
    );
    res.json({ message: "Documentation updated", repo: match.properties.name });
  } catch (error) {
    console.error("[update_docs] Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function get_docs(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  try {
    const repos = await db.get_repositories();
    const docs: Array<Record<string, { documentation: string }>> = [];

    let reposToQuery = repos;
    if (repoUrl) {
      const repoName = repoUrl.split("/").pop()?.replace(".git", "") || "";
      reposToQuery = repos.filter((repo) => {
        const matchesSourceLink = repo.properties.source_link === repoUrl;
        const matchesName = repo.properties.name === repoName;
        return matchesSourceLink || matchesName;
      });
    }

    for (const repo of reposToQuery) {
      if (repo.properties.documentation) {
        docs.push({
          [repo.properties.name]: {
            documentation: repo.properties.documentation,
          },
        });
      }
    }
    res.json(docs);
  } catch (error) {
    console.error("[get_docs] Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
}
