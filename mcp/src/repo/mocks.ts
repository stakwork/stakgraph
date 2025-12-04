import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { setBusy } from "../busy.js";
import { db } from "../graph/neo4j.js";

export interface MockInfo {
  name: string;
  files: string[];
  description: string;
}

export interface MocksResult {
  config: string[];
  mocks: MockInfo[];
}

export async function mocks_agent(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  if (!repoUrl) {
    res.status(400).json({ error: "Missing repo_url" });
    return;
  }
  const username = req.query.username as string | undefined;
  const pat = req.query.pat as string | undefined;
  const existingMocks = req.query.existing_mocks as string | undefined;

  const request_id = asyncReqs.startReq();
  try {
    cloneOrUpdateRepo(repoUrl, username, pat)
      .then(async (repoDir) => {
        console.log(`===> GET /mocks ${repoDir}`);
        let prompt = MOCKS_PROMPT;
        if (existingMocks) {
          const existingData = JSON.parse(existingMocks);
          prompt = buildIncrementalPrompt(existingData);
        }
        prompt += MOCKS_FINAL_ANSWER;
        const result = await get_context(
          prompt,
          repoDir,
          pat,
          undefined,
          MOCKS_SYSTEM
        );
        return result.content;
      })
      .then(async (result) => {
        const mocks = parseMocksResult(result);
        for (const mock of mocks.mocks) {
          console.log(
            `[mocks_agent] Discovered mock: ${mock.name} with files: ${mock.files.join(
              ", "
            )}`
          );
        }
        await persistMocksToGraph(mocks);
        asyncReqs.finishReq(request_id, mocks);
        setBusy(false);
        console.log("[mocks_agent] Background work completed, set busy=false");
      })
      .catch((error) => {
        console.error("[mocks_agent] Background work failed with error:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
        setBusy(false);
        console.log("[mocks_agent] Set busy=false after error");
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    console.error("Error in mocks_agent", error);
    res.status(500).json({ error: "Internal server error" });
    setBusy(false);
  }
}

async function persistMocksToGraph(mocksResult: MocksResult) {
  for (const mock of mocksResult.mocks) {
    try {
      const { ref_id } = await db.create_mock(
        mock.name,
        mock.description,
        mock.files
      );
      for (const filePath of mock.files) {
        await db.link_mock_to_file(ref_id, filePath);
      }
      for (const configPath of mocksResult.config) {
        await db.link_mock_to_file(ref_id, configPath);
      }
      console.log(`[mocks_agent] Created Mock node: ${mock.name} with ${mock.files.length} files`);
    } catch (e) {
      console.error(`[mocks_agent] Failed to create Mock node for ${mock.name}:`, e);
    }
  }
}

function buildIncrementalPrompt(existingMocks: MocksResult): string {
  return `I have already discovered some mocks in this codebase. Please briefly explore and see if you missed any mocks, or if there are additional integrations that need mocking.

EXISTING MOCKS:
${JSON.stringify(existingMocks, null, 2)}

Please search for any additional:
- HTTP requests to external services (not internal APIs)
- SDK method calls (AWS, Stripe, GitHub, etc.)
- Mock files or directories that weren't captured

`;
}

function parseMocksResult(content: string): MocksResult {
  try {
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }
    const parsed = JSON.parse(content);
    return parsed;
  } catch (e) {
    console.error("[mocks_agent] Failed to parse mocks result:", e);
    return { config: [], mocks: [] };
  }
}

const MOCKS_SYSTEM = `
You are a codebase exploration assistant. Your job is to identify 3rd party SERVICE integrations that may need mocking for testing or local development purposes.

IMPORTANT DISTINCTION:
- We want EXTERNAL SERVICES (APIs that talk to remote servers): Stripe API, AWS S3, GitHub API, SendGrid, Twilio, Pusher, etc.
- We do NOT want HTTP CLIENT LIBRARIES: axios, fetch, node-fetch, superagent, got, ky, etc.
- The goal is to mock the SERVICE, not the HTTP tool used to call it.

Look for:
1. SDK method calls - AWS SDK, Stripe SDK, GitHub API client, Twilio, SendGrid, Pusher, etc.
2. HTTP requests to SPECIFIC external domains (api.stripe.com, api.github.com, etc.)
3. Existing mock files or test doubles already in the codebase

Use the fulltext_search tool to find patterns like:
- SDK imports like "aws-sdk", "@aws-sdk", "stripe", "@octokit", "pusher", "twilio", etc.
- Mock directories like "__mocks__", "mocks/", "test/mocks", "api/mock"
- Environment variables that configure external services (API keys, service URLs)

DO NOT include:
- axios, fetch, node-fetch, superagent, got, ky (these are HTTP libraries, not services)
- Internal API calls between services in the same codebase

Be thorough but focused on external SERVICE integrations ONLY.
`;

const MOCKS_PROMPT = `
Please explore this codebase and identify all 3rd party integrations that may need mocking for local develop and/or testing.

Find:
1. HTTP requests to external services (not our own APIs)
2. SDK method calls (AWS, Stripe, GitHub, Twilio, etc.)
3. Any existing mock files or directories

`;

const MOCKS_FINAL_ANSWER = `
Return a JSON object with the following structure. Output ONLY the JSON in a code block, NOTHING ELSE:

\`\`\`json
{
  "config": ["path/to/config/file.ts", "path/to/env/config.ts"],
  "mocks": [
    {
      "name": "github",
      "files": ["src/lib/github.ts", "api/mock/github/route.ts"],
      "description": "GitHub API integration for repository operations"
    },
    {
      "name": "stripe",
      "files": ["src/payments/stripe.ts"],
      "description": "Stripe payment processing"
    },
    {
      "name": "sendgrid",
      "files": ["src/email/sender.ts"],
      "description": "SendGrid email service via HTTP API"
    }
  ]
}
\`\`\`

Rules:
- "config" should list files that contain configuration for external services (env vars, API keys, etc.)
- "files" should include both the integration file AND any existing mock files if found
- "description" should briefly explain what the integration does
- Only include EXTERNAL integrations, not internal API calls
`;
