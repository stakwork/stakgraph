import { CONFIG } from "./config.ts";
import { api } from "./routes/api.ts";

export function startServer() {
  console.log(`Starting server on port ${CONFIG.PORT}`);

  // Simulate server loop
  setInterval(() => {
    // Keep process alive for mock purposes
  }, 10000);
}

// Simple request handler simulation
export async function handleRequest(path: string, method: string, body?: any) {
  if (path === "/users" && method === "POST") {
    return await api.users.createUser(body);
  }
  if (path.startsWith("/users/") && method === "GET") {
    const id = path.split("/")[2];
    return await api.users.getUser(id);
  }
  return { error: "Not Found" };
}
