// EXAMPLE: Replace this directory with a real repo.
// Directory name format: {owner}--{repo}
// e.g. stakwork--hive/
//
// This file should contain the known-good pm2.config.js output.

module.exports = {
  apps: [
    {
      name: "frontend",
      script: "npm run dev",
      cwd: "/workspaces/repo",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PORT: "3000",
        INSTALL_COMMAND: "npm install",
      }
    }
  ],
};
