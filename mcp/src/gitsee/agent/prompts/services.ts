export const FILE_LINES = 100;

export const EXPLORER = `
You are a codebase exploration assistant. Your job is to identify the various services, integrations, and environment variables need to setup and run this codebase. Take your time exploring the codebase to find the most likely setup services, and env vars. You might need to use the fulltext_search tool to find instance of "process.env." or other similar patterns, based on the coding language(s) used in the project. If this is an Android project, also find Android-specific build/runtime details (AndroidManifest.xml, build.gradle/build.gradle.kts, gradlew, BuildConfig, ANDROID_ variables, adb usage). You will be asked to output actual configuration files at the end, so make sure you find everything you need to do that! Important: name one of the pm2 services "frontend" no matter what!!!
`;

export const SETUP_HINTS: Record<string, string> = {
  android: `
Android hint:
- The "frontend" service script should just be a placeholder echo; real work happens via env commands.
- Use PORT "8000" for the Android screen stream port via ws-scrcpy.
- Include PRE_START_COMMAND, INSTALL_COMMAND, BUILD_COMMAND, POST_START_COMMAND, REBUILD_COMMAND, and RESTART when Android files support them.
- PRE_START_COMMAND should ensure adb is ready, such as "adb wait-for-device".
- INSTALL_COMMAND should download Gradle dependencies, such as "./gradlew dependencies".
- BUILD_COMMAND should compile the APK, such as "./gradlew assembleDebug".
- POST_START_COMMAND should install and launch the APK using the package name and main activity from AndroidManifest.xml or build.gradle.
- REBUILD_COMMAND should match BUILD_COMMAND, and RESTART should be "true".
- The docker-compose.yml must use this fixed app/redroid structure. Add other services only if the app requires them:

\`\`\`yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ../..:/workspaces:cached
      - adb-keys:/root/.android
    ports:
      - "8000:8000"
      - "4724:4724"
    networks:
      - app_network
    depends_on:
      - redroid

  redroid:
    image: redroid/redroid:14.0.0-latest
    container_name: redroid
    privileged: true
    networks:
      - app_network
    volumes:
      - redroid-data:/data
    command:
      - androidboot.redroid_width=1080
      - androidboot.redroid_height=1920
      - androidboot.redroid_fps=30

networks:
  app_network:
    driver: bridge

volumes:
  adb-keys:
  redroid-data:
\`\`\`
`,
  elasticsearch: `
Elasticsearch hint:
- Add an elasticsearch docker-compose service using a single-node dev configuration.
- App env vars that point to Elasticsearch should use localhost and the exposed port.
`,
  livekit: `
LiveKit hint:
- Add a local livekit/livekit-server docker-compose service in dev mode.
- Use LIVEKIT_URL "ws://localhost:7880", LIVEKIT_API_KEY "devkey", and LIVEKIT_API_SECRET "devsecret-change-in-production-12345678901234".
`,
  mysql: `
MySQL hint:
- Add a mysql docker-compose service with deterministic dev database, user, and password.
- App env vars that point to MySQL should use localhost and the exposed port.
`,
  nextjs: `
Next.js hint:
- Use the repo package manager's dev command.
- Bind to all interfaces with the supported host flag, usually "-- --hostname 0.0.0.0".
- Prefer dev over start when a dev script exists.
`,
  postgres: `
Postgres hint:
- Add a postgres docker-compose service with deterministic POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD.
- App DATABASE_URL values should use localhost and the exposed port, not the compose service name.
- For Prisma apps, use "sleep 15 && npx prisma generate && npx prisma migrate deploy" for startup migrations; avoid "prisma migrate dev".
`,
  rabbitmq: `
RabbitMQ hint:
- Add a rabbitmq docker-compose service.
- App AMQP/RabbitMQ env vars should use localhost and the exposed port.
`,
  redis: `
Redis hint:
- Add a redis docker-compose service.
- App REDIS_URL values should use localhost and the exposed port, not the compose service name.
`,
  s3_compatible_storage: `
S3/R2/GCS-compatible storage hint:
- Add a MinIO docker-compose service so storage works locally in the pod.
- Expose MinIO S3 API on port 9000 and console on port 9001.
- Use access key "minioadmin" and secret key "minioadmin".
- Use the repo's bucket env var value when found; otherwise use "dev-bucket".
- Map repo env vars like R2_ENDPOINT_URL, S3_ENDPOINT, AWS_ENDPOINT_URL, or STORAGE_ENDPOINT to "http://127.0.0.1:9000".
- Map repo env vars like R2_ACCESS_KEY_ID, AWS_ACCESS_KEY_ID, or S3_ACCESS_KEY to "minioadmin".
- Map repo env vars like R2_SECRET_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, or S3_SECRET_KEY to "minioadmin".
- Create the bucket with a minio-init docker-compose service that depends on the MinIO healthcheck; avoid relying on app PRE_START_COMMAND timing for bucket creation.
- Prefer 127.0.0.1 over localhost for S3-compatible endpoints to avoid bucket.localhost DNS issues in SDKs that default to virtual-hosted bucket URLs.
`,
  supabase: `
Supabase hint:
- Use the Supabase CLI local stack for local Supabase projects.
- If the repo has supabase/migrations/*.sql, set PRE_START_COMMAND to "npx -y supabase start && npx -y supabase db reset --local" so the local database schema is applied.
- If the repo needs Supabase but has no supabase/config.toml, include "npx -y supabase init --force" before start.
- If schema SQL exists outside supabase/migrations, copy the repo-evidenced SQL file into supabase/migrations with a timestamped filename before running db reset. Do not assume a fixed source path.
- Do not use "supabase db query -f" for multi-statement schema files; prefer "supabase db reset --local".
- If no local schema or migration file is found, only start Supabase and do not invent a migration.
- Supabase CLI commands must run from the repo root (where supabase/ lives). If the service cwd is a subdirectory, prefix PRE_START_COMMAND with "cd /workspaces/MY_REPO_NAME &&".
- Use SUPABASE_URL / NEXT_PUBLIC_SUPABASE_URL value "http://localhost:54321".
- Use publishable key "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH" for browser/public Supabase env vars.
- Use secret key "sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz" for server-only Supabase secret env vars.
- Map these values to the exact Supabase env var names found in the repo.
`,
  vite: `
Vite hint:
- Use the repo package manager's dev command.
- Bind to all interfaces with "-- --host 0.0.0.0".
- Vite projects use dev, not start.
`,
};

const SETUP_HINT_IDS = Object.keys(SETUP_HINTS).sort().join(", ");

export const SETUP_PROFILER = `
You are a setup profiler. Explore the repo enough to identify the package manager, setup hints, and services needed to generate pm2.config.js and docker-compose.yml.

Only include things you actually find evidence for. Do not invent categories. Env vars alone show an integration exists; they do not prove a local service is required. Prefer known setup hint IDs: ${SETUP_HINT_IDS}.
`;

export const SETUP_PROFILE = `
Return a compact setup profile for this repo. Do not generate pm2.config.js or docker-compose.yml in this step.

Use only known values you can support from repo evidence. Put framework and dependency IDs in setup_hints, using only these values: ${SETUP_HINT_IDS}. If a field has no known values, return an empty array.

Put only compose/local services required for boot or core local flows in required_local_services, using setup hint IDs when possible.
`;

export const SETUP_PROFILE_SCHEMA = {
  type: "object",
  properties: {
    package_manager: { type: "string" },
    setup_hints: { type: "array", items: { type: "string" } },
    required_local_services: { type: "array", items: { type: "string" } },
    services: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          cwd: { type: "string" },
          framework: { type: "string" },
          dev_script: { type: "string" },
          port: { type: "number" },
        },
        required: ["name", "cwd"],
      },
    },
  },
  required: ["package_manager", "setup_hints", "required_local_services", "services"],
};

export function selectSetupHints(keys: Iterable<string>): string {
  const selected = Array.from(new Set(Array.from(keys).map((key) => key.toLowerCase())))
    .filter((key) => SETUP_HINTS[key])
    .sort();

  if (selected.length === 0) {
    return "SELECTED SETUP HINTS: none";
  }

  return [
    "SELECTED SETUP HINTS (use these when the dependency is required by repo env vars and services):",
    ...selected.map((key) => SETUP_HINTS[key].trim()),
  ].join("\n\n");
}

export const FINAL_ANSWER = `
Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION.

Return 2 files: a pm2.config.js and a docker-compose.yml. For each file, put "FILENAME: " followed by the filename (no markdown headers, just the plain filename), then the content in backticks. YOU MUST RETURN BOTH FILES!!!

# Pod contract - mandatory

The generated setup must run inside our pod/code-server environment. The repo is mounted at /workspaces/MY_REPO_NAME, so every pm2 cwd must start with /workspaces/MY_REPO_NAME. Unless a selected setup hint provides a platform-specific compose structure, docker-compose.yml must include the exact app service shown below. Do not replace it with a generic compose service. The app service is the long-running container where the repo code is mounted and pm2 commands run.

# Local-only dependency policy

The setup must boot without external accounts or real cloud credentials. For every required external dependency, use a local docker service, local emulator, self-hosted dev stack, placeholder, or deterministic dummy value.

- Use the selected setup hints injected above this prompt for dependency-specific setup.
- Only add docker-compose services required for the app to boot or core local flows to work. For optional integrations, use repo-supported mocks, disabled flags, or deterministic placeholder env vars.
- Do not emit cloud placeholders like your-project.supabase.co, Cloudflare R2 URLs, Stripe keys, Resend keys, Clerk keys, Auth0 domains, or LLM API keys as blockers.

# pm2.config.js

The actual dev services for running this project (MY_REPO_NAME). Often its just one single service! But sometimes the backend/frontend might be separate services. Each service env should have a INSTALL_COMMAND so our sandbox system knows how to install dependencies! You can also add optional BUILD_COMMAND, TEST_COMMAND, E2E_TEST_COMMAND, PRE_START_COMMAND, and POST_START_COMMAND if you find those in the package file. (an example of a PRE_START_COMMAND is a db migration script). Please name one of the services "frontend" no matter what!!!

The cwd should start with /workspaces/MY_REPO_NAME. For instance, if the frontend is within an "app" sub-directory in the repo, the cwd should be "/workspaces/MY_REPO_NAME/app". If the project is only a backend api, its fine to use the api service as the "frontend"... the "frontend" service is really just used to help our system identify whether things are running smoothly. IMPORTANT: if no frontend is found in this repo at all, add a dummy frontend service with "npx -y hell0-w0rld"! That is a simple hello world server that will run on port 3000, and will help the system check to see things are up and running. IMPORTANT: include other environment variables needed to run the project in the "env" section of the service!

### Other reminders for pm2.config.js:

1. **name MUST be "frontend"** — always use "frontend" as the app name, regardless of the repo name.

2. **Host binding** — the dev server MUST bind to 0.0.0.0:
  - Next.js: use the supported host flag, usually "-- --hostname 0.0.0.0" (e.g. "npm run dev -- --hostname 0.0.0.0")
  - Vite: append "-- --host 0.0.0.0" to the dev command (e.g. "npm run dev -- --host 0.0.0.0")
  - This flag is MANDATORY. Never omit it.

3. **Package manager** — the script must match INSTALL_COMMAND. pnpm → "pnpm run dev -- ...". yarn → "yarn dev -- --host 0.0.0.0" (for Vite) or "yarn dev -- --hostname 0.0.0.0" (for Next.js). Never use "npm run ..." if INSTALL_COMMAND is not npm.

4. **Use "dev" not "start"** — always prefer the dev script. Vite projects always have a "dev" script; never use "start" for Vite. Only fall back to "start" if package.json has no "dev" script at all.

5. **Environment variables**:
  - For empty/placeholder values, generate realistic dev defaults
  - For secrets: "dev-secret-key-change-in-production-12345678901234567890"
  - For hex encryption keys: "bb54aa41a75298418586c5443f264338013520b3bad612fce9ac2fc32ed19882"
  - Database URLs must match the credentials in docker-compose.yml

# docker-compose.yml

The auxiliary services needed to run the project, such as databases, caches, queues, etc.

Unless a selected setup hint provides a platform-specific compose structure, include this exact "app" service in docker-compose.yml:
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
      script: "npm run dev -- --hostname 0.0.0.0",
      cwd: "/workspaces/MY_REPO_NAME",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: "3000",
        INSTALL_COMMAND: "npm install",
        BUILD_COMMAND: "npm run build",
        DATABASE_URL: "postgresql://postgres:password@localhost:5432/backend_db"
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
