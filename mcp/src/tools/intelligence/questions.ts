import { decomposeQuestion, ask_question } from "./ask.js";
import { Answer } from "./ask.js";

export async function decomposeAndAsk(
  prompt: string,
  threshold: number,
  provider?: string,
  apiKey?: string
): Promise<Answer[]> {
  const answers: Answer[] = [];
  const dq = await decomposeQuestion(prompt, provider, apiKey);

  dq.questions.forEach((q) => {
    console.log(" ======== Q:", q);
  });
  for (const q of dq.questions) {
    const answer = await ask_question(q, threshold, provider, prompt, undefined, apiKey);
    if (answers.find((a) => a.hint_ref_id === answer.hint_ref_id)) {
      continue;
    }
    answers.push(answer);
  }
  return answers;
}

export const QUESTIONS = [
  "What is the overall architecture of this application (monolithic, microservices, serverless)?",
  "How does authentication work in this repo?",
  "What database technology is used and how is it configured?",
  "How are data models configured, is there an ORM being used?",
  "What frontend framework or library is being used (React, Vue, Angular, etc.)?",
  "What backend framework is being used (Express, Django, Rails, etc.)?",
  "How is the project structured and organized into folders/modules?",
  "What are the main entry points for the frontend and backend?",
  "How does authorization work - what permission system is in place?",
  "What API design pattern is used (REST, GraphQL, RPC)?",
  "How is routing handled on both frontend and backend?",
  "What state management solution is used on the frontend?",
  "How are environment variables and configuration managed?",
  "What testing frameworks and strategies are used?",
  "How is the application deployed (Docker, cloud services, traditional servers)?",
  "What build tools and bundlers are configured?",
  "How is database migration handled?",
  "What caching strategies are implemented?",
  "How are static assets handled and served?",
  "What error handling patterns are used throughout the application?",
  "How is logging implemented and where are logs stored?",
  "What monitoring and observability tools are in place?",
  "How are API requests validated on the backend?",
  "What security measures are implemented (CORS, CSRF, rate limiting)?",
  "How is user session management handled?",
  "What third-party services or APIs are integrated?",
  "How are database queries optimized and structured?",
  "What package managers and dependency management approaches are used?",
  "How is file upload and storage handled?",
  "What CI/CD pipelines are configured?",
  "How are database seeds or fixtures managed for development?",
  "What styling approach is used (CSS-in-JS, SASS, CSS modules, etc.)?",
  "How is email functionality implemented? Is there any email integration?",
  "What search functionality exists and how is it implemented?",
  "How are real-time features handled (WebSockets, Server-Sent Events)?",
  "What background job processing system is used?",
  "What internationalization (i18n) support exists?",
  "How are forms handled and validated on the frontend?",
  "What accessibility considerations are implemented?",
  "How is performance monitoring and optimization handled?",
  "What security scanning or vulnerability checking is in place?",
  "How are database relationships and constraints defined?",
  "What code formatting and linting rules are enforced?",
  "How is component or module reusability achieved?",
  "What design system or UI component library is used?",
  "How are different environments (dev, staging, prod) managed?",
  "What backup and disaster recovery procedures exist?",
  "How is API documentation generated and maintained?",
];
