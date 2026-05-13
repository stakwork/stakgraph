// Create a single virtual key for ad-hoc testing.
//
// Usage:
//   tsx create-vk.ts [name]
//
// Prints the VK value (sk-bf-...) on stdout so it can be captured by other scripts:
//   VK=$(tsx create-vk.ts test-run-1)

import { createVirtualKey, ping } from "./lib.js";

async function main() {
  await ping();
  const name = process.argv[2] ?? `test-vk-${Date.now()}`;
  const vk = await createVirtualKey({
    name,
    description: "Created by mcp/docs/gateway/create-vk.ts",
    budget: { max_limit: 5.0, reset_duration: "1d" },
    rate_limit: {
      request_max_limit: 60,
      request_reset_duration: "1m",
    },
    // No provider_configs => any configured provider/model is allowed.
  });
  // Emit the VK details to stderr, the value alone to stdout.
  console.error(JSON.stringify({ id: vk.id, name: vk.name }, null, 2));
  console.log(vk.value);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
