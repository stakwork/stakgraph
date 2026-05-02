import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { startTracking, endTracking } from "../busy.js";
import { parse_files_contents } from "../gitsee/agent/index.js";
import { randomUUID } from "crypto";
import { createSession, appendSessionEnd } from "./session.js";
import { getModelDetails } from "../aieo/src/index.js";
import {
  EXPLORER,
  FINAL_ANSWER as GITSEE_FINAL_ANSWER,
  SETUP_PROFILE,
  SETUP_PROFILER,
  SETUP_PROFILE_SCHEMA,
} from "../gitsee/agent/prompts/services.js";
import {
  SetupProfile,
  buildRepoFacts,
  buildSelectedHints,
  combineUsage,
} from "./services_utils.js";

// curl "http://localhost:3355/progress?request_id=123"
export async function services_agent(req: Request, res: Response) {
  const owner = req.query.owner as string;
  const repoName = req.query.repo as string | undefined;
  if (!repoName || !owner) {
    res.status(400).json({ error: "Missing repo" });
    return;
  }
  const username = req.query.username as string | undefined;
  const pat = req.query.pat as string | undefined;

  const request_id = asyncReqs.startReq();
  const sessionId = randomUUID();
  const startTime = Date.now();
  const { modelId, provider } = getModelDetails();
  createSession(sessionId, undefined, "services_agent");
  const opId = startTracking("services_agent");
  try {
    cloneOrUpdateRepo(`https://github.com/${owner}/${repoName}`, username, pat)
      .then(async (repoDir) => {
        console.log(`===> services_agent cloned ${repoDir}`);

        const repoFacts = await buildRepoFacts(repoDir);

        const profilePrompt = [
          repoFacts.factsBlock,
          "How do I set up this repo? I want to run the project on my remote code-server environment. Please prioritize web services that I will be able to run there (so ignore fancy stuff like web extension, desktop app using electron, etc). Lets just focus on bare-bones setup to install, build, and run a web frontend, and supporting services like the backend.",
          SETUP_PROFILE,
        ].join("\n\n");

        const profileResult = await get_context(profilePrompt, repoDir, {
          pat,
          systemOverride: SETUP_PROFILER,
          schema: SETUP_PROFILE_SCHEMA,
          sessionId,
          source: "services_agent_profile",
        });

        const setupProfile = profileResult.content as SetupProfile;
        const selectedHints = buildSelectedHints(setupProfile);

        const fad = GITSEE_FINAL_ANSWER.replaceAll("MY_REPO_NAME", repoName);
        const prompt =
          "SETUP PROFILE (use this as ground truth for framework and dependency selection):\n" +
          JSON.stringify(setupProfile, null, 2) +
          "\n\n" +
          selectedHints +
          "\n\n" +
          "Use required_local_services from the setup profile as the source of truth for docker-compose services. Do not add compose services for optional integrations that are not listed there." +
          "\n\n" +
          "How do I set up this repo? I want to run the project on my remote code-server environment. Please prioritize web services that I will be able to run there (so ignore fancy stuff like web extension, desktop app using electron, etc). Lets just focus on bare-bones setup to install, build, and run a web frontend, and supporting services like the backend." +
          "\n\nIMPORTANT: In pm2.config.js, any env var that points to a docker-compose service (e.g. DATABASE_URL, REDIS_URL) MUST use 'localhost' as the hostname, NOT the container service name. The 'app' container has extra_hosts configured so localhost resolves correctly to the host bridge. Never use the docker service name (e.g. 'postgres', 'redis') as a hostname." +
          "\n\n" +
          fad;

        const text_of_files = await get_context(prompt, repoDir, {
          pat,
          systemOverride: EXPLORER,
          sessionId,
          source: "services_agent",
        });

        const files = parse_files_contents(text_of_files.content);

        return { files, usage: combineUsage(profileResult.usage, text_of_files.usage) };
      })
      .then(async (result) => {
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: "success",
        });
        asyncReqs.finishReq(request_id, {
          ...result.files,
          usage: result.usage,
          sessionId,
        });
      })
      .catch(async (error) => {
        console.error("[services_agent] Background work failed with error:", error);
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: "error",
          error_message: error.message || error.toString(),
        });
        asyncReqs.failReq(request_id, error.message || error.toString());
      })
      .finally(() => {
        endTracking(opId);
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    console.error("Error in services_agent", error);
    res.status(500).json({ error: "Internal server error" });
    endTracking(opId);
  }
}
