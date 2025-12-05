import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { setBusy } from "../busy.js";
import { db } from "../graph/neo4j.js";
import { z } from "zod";

export interface MockInfo {
  name: string;
  files: string[];
  description: string;
  mocked: boolean;
}

export interface ExistingMockInfo {
  name: string;
  description: string;
  mocked: boolean;
}

export interface MocksResult {
  config: string[];
  mocks: MockInfo[];
}

const MockInfoSchema = z.object({
  name: z.string(),
  files: z.array(z.string()),
  description: z.string(),
  mocked: z.boolean(),
});

const MocksResultSchema = z.object({
  config: z.array(z.string()),
  mocks: z.array(MockInfoSchema),
});

export async function mocks_agent(req: Request, res: Response) {
  const repoUrl = req.query.repo_url as string;
  if (!repoUrl) {
    res.status(400).json({ error: "Missing repo_url" });
    return;
  }
  const username = req.query.username as string | undefined;
  const pat = req.query.pat as string | undefined;
  const sync = req.query.sync === "true" || req.query.sync === "1";

  const request_id = asyncReqs.startReq();
  try {
    cloneOrUpdateRepo(repoUrl, username, pat)
      .then(async (repoDir) => {
        console.log(`===> GET /mocks ${repoDir} (sync=${sync})`);
        const existingMocksNodes = await db.get_all_mocks();
        let prompt = MOCKS_PROMPT;
        if (existingMocksNodes.length > 0) {
          const minimalMocks: ExistingMockInfo[] = existingMocksNodes.map(
            (m) => ({
              name: m.properties.name || "",
              description: m.properties.description || "",
              mocked: m.properties.mocked ?? false,
            })
          );
          prompt = buildIncrementalPrompt(minimalMocks, sync);
        }
        prompt += MOCKS_FINAL_ANSWER;
        const schema = z.toJSONSchema(MocksResultSchema);
        const result = await get_context(
          prompt,
          repoDir,
          pat,
          undefined,
          MOCKS_SYSTEM,
          schema
        );
        // When schema is provided, result.content is already the structured object
        return result.content as MocksResult;
      })
      .then(async (mocks: MocksResult) => {
        console.log(`[mocks_agent] Parsed result:`, JSON.stringify(mocks, null, 2));
        for (const mock of mocks.mocks) {
          console.log(
            `[mocks_agent] Discovered mock: ${
              mock.name
            } with files: ${mock.files.join(", ")}`
          );
        }
        await persistMocksToGraph(mocks);
        asyncReqs.finishReq(request_id, mocks);
        setBusy(false);
        console.log("[mocks_agent] Background work completed, set busy=false");
      })
      .catch((error) => {
        console.error(
          "[mocks_agent] Background work failed with error:",
          error
        );
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
        mock.files,
        mock.mocked
      );
      for (const filePath of mock.files) {
        await db.link_mock_to_file(ref_id, filePath);
      }
      for (const configPath of mocksResult.config) {
        await db.link_mock_to_file(ref_id, configPath);
      }
      console.log(
        `[mocks_agent] Created Mock node: ${mock.name} with ${mock.files.length} files`
      );
    } catch (e) {
      console.error(
        `[mocks_agent] Failed to create Mock node for ${mock.name}:`,
        e
      );
    }
  }
}

function buildIncrementalPrompt(existingMocks: ExistingMockInfo[], sync: boolean = false): string {
  const syncInstructions = sync
    ? `
IMPORTANT - SYNC MODE:
This is an incremental sync. Focus on:
1. NEW services that weren't in the existing mocks list
2. CHANGED mock status - verify if any services now have mock implementations (or lost them)
   - If a service was "mocked": false but now has mock files, return it with "mocked": true
   - If a service was "mocked": true but mock files are gone, return it with "mocked": false
3. Only return mocks that are NEW or have CHANGED status

Do NOT return services that already exist with the same mock status.
`
    : `
Please search for any additional:
- HTTP requests to external services (not internal APIs)
- SDK method calls (AWS, Stripe, GitHub, etc.)
- Mock files or directories that weren't captured
`;

  return `I have already discovered some mocks in this codebase. Please briefly explore and see if you missed any mocks, or if there are additional integrations that need mocking.

EXISTING MOCKS:
${JSON.stringify(existingMocks, null, 2)}
${syncInstructions}
`;
}

const MOCKS_SYSTEM = `
You are a codebase exploration assistant. Your job is to identify 3rd party SERVICE integrations and determine if they already have mock implementations.

CRITICAL DISTINCTION - What counts as a "service":
YES - Include these (calling THEIR API endpoint over HTTP/network):
   - Stripe API, GitHub API, AWS S3, SendGrid, Twilio, Pusher, Firebase
   - Payment processors, email services, cloud storage, SMS providers
   - Third-party REST/GraphQL APIs accessed over network

NO - Exclude these (using THEIR CODE/LIBRARY locally):
   - HTTP client libraries: axios, fetch, node-fetch, superagent, got, ky, request
   - Auth libraries: NextAuth, Passport, Auth0 SDK, OAuth libraries
   - WebSocket/realtime clients: Socket.io client, @anycable/web, Pusher-js client
   - State management: Redux, Zustand, Jotai, Recoil
   - UI component libraries: Material-UI, Chakra, shadcn/ui
   - Utility libraries: lodash, date-fns, moment, validator
   - ORM/database clients used locally: Prisma, TypeORM, Mongoose (unless calling a hosted DB service)

RULE OF THUMB: If you're calling THEIR API endpoint = service. If you're importing THEIR NPM/gem/pip package = NOT a service.

EXAMPLES:
correct: "@stripe/stripe-js" making API calls to api.stripe.com = SERVICE (include it)
wrong: "next-auth" for OAuth flows = LIBRARY (exclude it)
correct: "aws-sdk" uploading to S3 = SERVICE (include it)
wrong: "socket.io-client" for WebSocket connections = LIBRARY (exclude it)
correct: "twilio" sending SMS via API = SERVICE (include it)
wrong: "@anycable/web" for WebSocket client = LIBRARY (exclude it)

Look for:
1. SDK method calls that make network requests - AWS SDK, Stripe SDK, GitHub API client, Twilio, SendGrid, Firebase Admin
2. HTTP requests to SPECIFIC external domains (api.stripe.com, api.github.com, api.openai.com)
3. Existing mock files or test doubles already in the codebase

For EACH service you find, determine if it has an existing mock by checking:
- Mock files in __mocks__/, mocks/, test/mocks, api/mock directories
- Jest mocks, test doubles, or stub implementations
- Fake/mock classes or functions in test files

Use the fulltext_search tool to find patterns like:
- SDK imports like "aws-sdk", "@aws-sdk", "stripe", "@octokit", "twilio", "@sendgrid/mail"
- Mock directories like "__mocks__", "mocks/", "test/mocks", "api/mock"
- Environment variables that configure external services (API keys, service URLs)

DO NOT include:
- HTTP client libraries (axios, fetch, etc.)
- Auth/OAuth frameworks (NextAuth, Passport, Auth0 SDK)
- WebSocket/realtime client libraries (Socket.io-client, @anycable/web, Pusher-js)
- State management, UI libraries, utilities
- Internal API calls between services in the same codebase

Be thorough but focused on external SERVICE integrations that make network calls to third-party APIs.
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
      "description": "GitHub API integration for repository operations",
      "mocked": true
    },
    {
      "name": "stripe",
      "files": ["src/payments/stripe.ts"],
      "description": "Stripe payment processing",
      "mocked": false
    },
    {
      "name": "sendgrid",
      "files": ["src/email/sender.ts"],
      "description": "SendGrid email service via HTTP API",
      "mocked": false
    }
  ]
}
\`\`\`

Rules:
- "config" should list files that contain configuration for external services (env vars, API keys, etc.)
- "files" should include both the integration file AND any existing mock files if found
- "description" should briefly explain what the integration does
- "mocked" should be true if you found an existing mock/stub/fake implementation, false otherwise
- Only include EXTERNAL integrations, not internal API calls
`;
