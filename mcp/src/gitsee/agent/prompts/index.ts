export * as first_pass from "./first_pass.js";
export * as features from "./features.js";
export * as services from "./services.js";

const generic = {
  FILE_LINES: 80,
  EXPLORER: `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.`,
  FINAL_ANSWER: `Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION.`,
};

export { generic };
