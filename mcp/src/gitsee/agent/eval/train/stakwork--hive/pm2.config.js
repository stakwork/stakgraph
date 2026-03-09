module.exports = {
  apps: [
    {
      name: "frontend",
      script: "npm run dev -- -H 0.0.0.0",
      cwd: "/workspaces/hive",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        DATABASE_URL:
          "postgresql://hive_user:hive_password@localhost:5432/hive_db",
        NEXTAUTH_URL: "http://localhost:3000",
        NEXTAUTH_SECRET:
          "dev-secret-key-change-in-production-12345678901234567890",
        JWT_SECRET:
          "dev-jwt-secret-change-in-production-1234567890abcdefghijklmnopqrstuvwxyz",
        POD_URL: "http://localhost:3000",
        TOKEN_ENCRYPTION_KEY:
          "bb54aa41a75298418586c5443f264338013520b3bad612fce9ac2fc32ed19882",
        TOKEN_ENCRYPTION_KEY_ID: "k2",
        API_TOKEN: "dev-api-token-change-in-production",
        USE_MOCKS: "true",
        GITHUB_CLIENT_ID: "mock-github-client-id",
        GITHUB_CLIENT_SECRET: "mock-github-client-secret",
        GITHUB_APP_SLUG: "mock-github-app",
        STAKWORK_API_KEY: "mock-stakwork-api-key",
        STAKWORK_BASE_URL: "https://jobs.stakwork.com/api/v1",
        STAKWORK_WORKFLOW_ID: "43198,43199,43200",
        ANTHROPIC_API_KEY: "mock-anthropic-api-key",
        NEXT_PUBLIC_SPHINX_TRIBES_URL: "https://community.sphinx.chat",
        PORT: "3000",
        INSTALL_COMMAND: "npm install",
        TEST_COMMAND: "npm run test",
        BUILD_COMMAND: "npm run build",
        PRE_START_COMMAND: "npx prisma migrate dev",
        RESET_COMMAND: "npx -y prisma migrate reset -f",
      },
    },
  ],
};
