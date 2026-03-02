export const FILE_LINES = 100;

export const EXPLORER = `
You are a codebase exploration assistant. Your job is to identify the various services, integrations, and environment variables need to setup and run this codebase. Take your time exploring the codebase to find the most likely setup services, and env vars. You might need to use the fulltext_search tool to find instance of "process.env." or other similar patterns, based on the coding language(s) used in the project. If this is an Android project, also find Android-specific build/runtime details (AndroidManifest.xml, build.gradle/build.gradle.kts, gradlew, BuildConfig, ANDROID_ variables, adb usage). You will be asked to output actual configuration files at the end, so make sure you find everything you need to do that! Important: name one of the pm2 services "frontend" no matter what!!!
`;

export const FINAL_ANSWER = `
Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION.

Return 2 files: a pm2.config.js and a docker-compose.yml. For each file, put "FILENAME: " followed by the filename (no markdown headers, just the plain filename), then the content in backticks. YOU MUST RETURN BOTH FILES!!!

- pm2.config.js: the actual dev services for running this project (MY_REPO_NAME). Often its just one single service! But sometimes the backend/frontend might be separate services. Each service env should have a INSTALL_COMMAND so our sandbox system knows how to install dependencies! You can also add optional BUILD_COMMAND, TEST_COMMAND, E2E_TEST_COMMAND, PRE_START_COMMAND, and POST_START_COMMAND if you find those in the package file. (an example of a PRE_START_COMMAND is a db migration script). Please name one of the services "frontend" no matter what!!! The cwd should start with /workspaces/MY_REPO_NAME. For instance, if the frontend is within an "app" sub-directory in the repo, the cwd should be "/workspaces/MY_REPO_NAME/app". If the project is only a backend api, its fine to use the api service as the "frontend"... the "frontend" service is really just used to help our system identify whether things are running smoothly. IMPORTANT: if no frontend is found in this repo at all, add a dummy frontend service with "npx -y hell0-w0rld"! That is a simple hello world server that will run on port 3000, and will help the system check to see things are up and running. IMPORTANT: include other environment variables needed to run the project in the "env" section of the service!

  If the project is Android:
  - The "frontend" service script should just be a placeholder echo — all real work happens via the env commands.
  - PORT should be "8000" (the Android screen stream port via ws-scrcpy).
  - PRE_START_COMMAND: ensure adb is ready (e.g. "adb wait-for-device").
  - INSTALL_COMMAND: download gradle dependencies (e.g. "./gradlew dependencies").
  - BUILD_COMMAND: compile the APK (e.g. "./gradlew assembleDebug").
  - POST_START_COMMAND: install the APK and launch the app (e.g. "adb install -r ... && adb shell am start -n ...").
  - REBUILD_COMMAND: recompile after code changes — same as BUILD_COMMAND (this is Android's hot reload equivalent).
  - RESTART: always "true" for Android so code changes trigger a rebuild + reinstall.
  - Find the correct package name and main activity from AndroidManifest.xml or build.gradle.

- docker-compose.yml: the auxiliary services needed to run the project, such as databases, caches, queues, etc.

  If this is NOT Android, there is a special "app" service in the docker-compose.yml that you MUST include! It is the service in which the codebase is mounted. Here is the EXACT content that it should have:
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

  If this IS Android, the docker-compose.yml has a fixed structure that MUST be followed exactly. The "app" and "redroid" services are mandatory and must not be modified. You may add extra services (postgres, redis, etc.) only if the Android app explicitly requires them.

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

  # Add other services here only if the app requires them (postgres, redis, etc.)

networks:
  app_network:
    driver: bridge

volumes:
  adb-keys:
  redroid-data:
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

# ANDROID EXAMPLE OUTPUT:

FILENAME: pm2.config.js

\`\`\`js
module.exports = {
  apps: [
    {
      name: "frontend",
      script: "adb install -r app/build/outputs/apk/debug/app-debug.apk && adb shell am start -n com.example.app/.MainActivity",
      cwd: "/workspaces/MY_REPO_NAME",
      instances: 1,
      autorestart: false,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: "8000",
        PRE_START_COMMAND: "adb wait-for-device",
        INSTALL_COMMAND: "./gradlew dependencies",
        BUILD_COMMAND: "./gradlew assembleDebug",
        POST_START_COMMAND: "bash -c 'echo \"App installed and launched via script\"'",
        RESTART: "true",
        REBUILD_COMMAND: "./gradlew assembleDebug",
      }
    }
  ],
};
\`\`\`

`;
