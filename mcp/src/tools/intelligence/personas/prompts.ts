const PM_PROMPT = `
You are a Product Manager.
`;

const SENIOR_DEV_PROMPT = `
You are a Senior Developer. Please rephrase the following question/answer pair as a Senior Developer, who already knows the codebase well. Focus on highlighting files, functions, pages, etc, that are relevant for implementing changes to the feature. Keep your question and answer concise and to the point! You don't need to include every general high-level aspect of the answer! Just a brief, insightful, specific recommendation.
`;

const JUNIOR_DEV_PROMPT = `
You are a Junior Developer. Please rephrase the following question/answer pair as a Junior Developer, who needs guidance in order to understand how the project works. Focus on highlighting files, functions, pages, etc, that are relevant for implementing changes to the feature.
`;

const CEO_PROMPT = `
You are the CEO of the company. Please rephrase the following question/answer pair as a CEO, who only needs the high-level overview of the feature. The answer you generate should be only a couple sentences! Distill the question/answer to the most important point!
`;

const AGENT_PROMPT = `
You are an AI agent. Please rephrase the following question/answer pair as an AI agent, who has no prior context of the codebase! Focus on highlighting files, functions, pages, etc, that are relevant for implementing changes to the feature. And to the best of your ability, refine the answer to best explain how the agent can 
`;

function mp(prompt: string, question: string, answer: string) {
  return `
  **Instructions:**
   ${prompt}

  **Question:**
  ${question}

  **Answer:**
  ${answer}

  **Output:**
  {
    "question": "...",
    "answer": "..."
  }
  `;
}

const PM = (q: string, a: string) => mp(PM_PROMPT, q, a);
const SENIOR_DEV = (q: string, a: string) => mp(SENIOR_DEV_PROMPT, q, a);
const JUNIOR_DEV = (q: string, a: string) => mp(JUNIOR_DEV_PROMPT, q, a);
const CEO = (q: string, a: string) => mp(CEO_PROMPT, q, a);
const AGENT = (q: string, a: string) => mp(AGENT_PROMPT, q, a);

export { PM, SENIOR_DEV, JUNIOR_DEV, CEO, AGENT };
