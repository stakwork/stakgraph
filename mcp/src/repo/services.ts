import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { cloneOrUpdateRepo } from "./clone.js";
import { get_context } from "./agent.js";
import { setBusy } from "../busy.js";
import { parse_files_contents } from "../gitsee/agent/index.js";

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
  setBusy(true);
  try {
    cloneOrUpdateRepo(`https://github.com/${owner}/${repoName}`, username, pat)
      .then(async (repoDir) => {
        console.log(`===> POST /repo/agent ${repoDir}`);
        // replace all instance of MY_REPO_NAME with the actual repo name
        const fad = FINAL_ANSWER.replaceAll("MY_REPO_NAME", repoName);
        const prompt =
          "How do I set up this repo? I want to run the project on my remote code-server environment. Please prioritize web services that I will be able to run there (so ignore fancy stuff like web extension, desktop app using electron, etc). Lets just focus on bare-bones setup to install, build, and run a web frontend, and supporting services like the backend." +
          fad;
        const text_of_files = await get_context(prompt, repoDir, {
          pat,
          systemOverride: SERVICES_SYSTEM,
        });
        return text_of_files.content;
      })
      .then((result) => {
        const files = parse_files_contents(result);
        asyncReqs.finishReq(request_id, files);
        setBusy(false);
        console.log("[repo_agent] Background work completed, set busy=false");
      })
      .catch((error) => {
        console.error("[repo_agent] Background work failed with error:", error);
        asyncReqs.failReq(request_id, error.message || error.toString());
        setBusy(false);
        console.log("[repo_agent] Set busy=false after error");
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    console.error("Error in repo_agent", error);
    res.status(500).json({ error: "Internal server error" });
    setBusy(false);
  }
}

const SERVICES_SYSTEM = `
You are a codebase exploration assistant. Your job is to identify the various services, integrations, and environment variables needed to setup and run this codebase. Take your time exploring the codebase to find the most likely setup services, and env vars. You might need to use the fulltext_search tool to find instance of "process.env." or other similar patterns, based on the coding language(s) used in the project. You will be asked to output actual configuration files at the end, so make sure you find everything you need to do that!
`;

const FINAL_ANSWER = `
Return two files: a pm2.config.js and a docker-compose.yml. For each file, put "FILENAME: " followed by the filename (no markdown headers, just the plain filename), then the content in backticks. YOU MUST RETURN 2 FILES!!!

- pm2.config.js: the actual dev services for running this project (MY_REPO_NAME). Often its just one single service! But sometimes the backend/frontend might be separate services. IMPORTANT: each service env should have a INSTALL_COMMAND so our sandbox system knows how to install dependencies! You can also add optional BUILD_COMMAND, TEST_COMMAND, E2E_TEST_COMMAND, and PRE_START_COMMAND env vars if you find those in the package file. (an example of a PRE_START_COMMAND is a db migration script). And of course add other env vars specific to the service. IMPORTANT: config env vars that point to other docker services (such as DATABASE_URL) can use "localhost", since the "app" container is using custom docker bridge network, and has extra_hosts configured. You SHOULD NOT reference the other container name as the hostname. Please name one of the services "frontend" no matter what. The cwd should start with /workspaces/MY_REPO_NAME. For instance, if the frontend is within a "frontend" dir, the cwd should be "/workspaces/MY_REPO_NAME/frontend".
- docker-compose.yml: the auxiliary services needed to run the project, such as databases, caches, queues, etc. IMPORTANT: there is a special "app" service in the docker-compsose.yaml that you MUST include! It is the service in which the codebase is mounted. Here is the EXACT content that it should have:
\`\`\`
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ../..:/workspaces:cached
    command: sleep infinity
    networks:
      - app_network
    extra_hosts:
      - "localhost:172.17.0.1"
      - "host.docker.internal:host-gateway"
\`\`\`

# HERE IS AN EXAMPLE OUTPUT:

FILENAME: pm2.config.js

\`\`\`js
module.exports = {
  apps: [
    {
      name: "frontend",
      script: "npm run dev",
      cwd: "/workspaces/MY_REPO_NAME",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: "3000",
        INSTALL_COMMAND: "npm install",
        BUILD_COMMAND: "npm run build",
        DATABASE_URL: "postgresql://postgres:password@localhost:5432/backend_db",
        JWT_KEY: "your_jwt_secret_key"
      }
    }
  ],
};
\`\`\`

FILENAME: docker-compose.yml

\`\`\`yaml
version: '3.8'
networks:
  app_network:
    driver: bridge
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ../..:/workspaces:cached
    command: sleep infinity
    networks:
      - app_network
    extra_hosts:
      - "localhost:172.17.0.1"
      - "host.docker.internal:host-gateway"
  postgres:
    image: postgres:15
    container_name: backend-postgres
    environment:
      - POSTGRES_DB=backend_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app_network
    restart: unless-stopped
volumes:
  postgres_data:
\`\`\`

`;
