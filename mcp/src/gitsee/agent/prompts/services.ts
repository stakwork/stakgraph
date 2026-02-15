export const FILE_LINES = 100;

export const EXPLORER = `
You are a codebase exploration assistant. Your job is to identify the various services, integrations, and environment variables need to setup and run this codebase. Take your time exploring the codebase to find the most likely setup services, and env vars. You might need to use the fulltext_search tool to find instance of "process.env." or other similar patterns, based on the coding language(s) used in the project. You will be asked to output actual configuration files at the end, so make sure you find everything you need to do that!
`;

export const FINAL_ANSWER = `
Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION.

Return three files: a pm2.config.js, a .env file, and a docker-compose.yml. For each file, put "FILENAME: " followed by the filename (no markdown headers, just the plain filename), then the content in backticks. YOU MUST RETURN ALL 3 FILES!!!

- pm2.config.js: the actual dev services for running this project (MY_REPO_NAME). Often its just one single service! But sometimes the backend/frontend might be separate services. IMPORTANT: each service env should have a INSTALL_COMMAND so our sandbox system knows how to install dependencies! You can also add optional BUILD_COMMAND, TEST_COMMAND, E2E_TEST_COMMAND, and PRE_START_COMMAND if you find those in the package file. (an example of a PRE_START_COMMAND is a db migration script). Please name one of the services "frontend" no matter what. The cwd should start with /workspaces/MY_REPO_NAME. For instance, if the frontend is within an "app" sub-directory in the repo, the cwd should be "/workspaces/MY_REPO_NAME/app". If the project is only a backend api, its fine to use the api service as the "frontend"... the "frontend" service is really just used to help our system identify whether things are running smoothly. IMPORTANT: if no frontend is found in this repo at all, you can add a dummy frontend with "npx -y hell0-w0rld"! That is a simple hello world server that will run on port 3000, and will help the system check to see things are up and running.
- .env: the environment variables needed to run the project, with example values.
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
        BUILD_COMMAND: "npm run build"
      }
    }
  ],
};
\`\`\`

FILENAME: .env

\`\`\`sh
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/backend_db
JWT_KEY=your_jwt_secret_key
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
