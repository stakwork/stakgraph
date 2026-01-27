import { handleRequest } from "../server.ts";

async function runTests() {
  console.log("Running integration tests...");

  // Test Create User
  const newUser = await handleRequest("/users", "POST", {
    username: "integration",
    email: "int@example.com",
  });

  if (newUser && newUser.id) {
    console.log("PASS: Create User API");
  } else {
    console.error("FAIL: Create User API");
  }
}

runTests();
