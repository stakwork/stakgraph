import { db } from "../src/graph/neo4j.js";
import { vectorizeCodeDocument } from "../src/vector/index.js";

const RETRY_COUNT = 10;
const RETRY_DELAY_MS = 2000;

async function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function findNodesWithRetry(name: string, type: string) {
  for (let i = 0; i < RETRY_COUNT; i++) {
    const nodes = await db.findNodesByName(name, type);
    if (nodes.length > 0) {
      return nodes;
    }
    console.log(
      `[seed-embeddings] Waiting for node '${name}' to appear (attempt ${i + 1}/${RETRY_COUNT})...`,
    );
    await wait(RETRY_DELAY_MS);
  }
  return [];
}

async function main() {
  console.log("[seed-embeddings] Starting...");

  const nodesToSeed = [
    {
      name: "App",
      type: "Function",
      description:
        "The main React application component that sets up the routing structure using `react-router-dom`. It defines routes for displaying the list of people at `/people` and creating a new person at `/new-person`.",
    },
    {
      name: "NewRouter",
      type: "Function",
      description:
        "Initializes the Chi router, sets up routes for Person endpoints (GET /person/{id}, POST /person, GET /people), and starts the HTTP server on the specified port.",
    },
    {
      name: "CreatePerson",
      type: "Function",
      description:
        "Handles HTTP POST requests to create a new Person entity by reading the request body, unmarshaling the JSON, and saving it to the database.",
    },
  ];

  try {
    for (const item of nodesToSeed) {
      console.log(
        `[seed-embeddings] Processing node: ${item.name} (${item.type})`,
      );

      const nodes = await findNodesWithRetry(item.name, item.type);

      if (nodes.length === 0) {
        console.warn(
          `[seed-embeddings] Could not find '${item.name}' node. Skipping.`,
        );
        continue;
      }

      const node = nodes[0];
      console.log(
        `[seed-embeddings] Found node: ${node.properties.name} (${node.ref_id})`,
      );

      console.log(
        `[seed-embeddings] Generating embeddings for description: "${item.description}"`,
      );
      const embeddings = await vectorizeCodeDocument(item.description);
      console.log(
        `[seed-embeddings] Generated embeddings (length: ${embeddings.length})`,
      );

      if (!node.ref_id) {
        throw new Error(`Node ${item.name} missing ref_id`);
      }

      await db.update_node_description_and_embeddings(
        node.ref_id,
        item.description,
        embeddings,
      );
      console.log(
        `[seed-embeddings] Node '${item.name}' updated successfully.`,
      );
    }
  } catch (error) {
    console.error(
      "[seed-embeddings] Error generating/updating embeddings:",
      error,
    );
    try {
      await db.close();
    } catch (e) {
      console.error("[seed-embeddings] Error closing db:", e);
    }
    process.exit(1);
  }

  try {
    await db.close();
    console.log("[seed-embeddings] Database connection closed.");
  } catch (e) {
    console.error("[seed-embeddings] Error closing db:", e);
  }
}

main();
