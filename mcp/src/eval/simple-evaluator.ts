import { z } from "zod";
import * as dotenv from "dotenv";
import { generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { openai } from "@ai-sdk/openai";
import { getOrCreateStagehand } from "../tools/stagehand/core.js";
import { Step, TestResult } from "./types.js";

// Load environment variables
dotenv.config();

// Define the schema for step generation
const StepGenerationSchema = z.object({
  steps: z.array(
    z.discriminatedUnion("type", [
      z.object({
        type: z.literal("navigate"),
        url: z.string(),
      }),
      z.object({
        type: z.literal("observe"),
        instruction: z.string(),
      }),
      z.object({
        type: z.literal("extract"),
        instruction: z.string(),
      }),
      z.object({
        type: z.literal("screenshot"),
        path: z.string(),
        fullPage: z.boolean().optional(),
      }),
    ])
  ),
  testCriteria: z.array(z.string()),
});

const SYSTEM_PROMPT = `You are an expert at creating browser automation test steps using Stagehand to verify specific criteria.

Your job is to create steps that will verify the given prompt. The base URL will be provided, so you only need to focus on the path and criteria verification.

Stagehand capabilities:
- navigate: Go to a specific URL
- observe: Find and identify page elements (use for element existence checks)
- extract: Extract text content from elements (use for content verification)
- screenshot: Take a screenshot of the current page

IMPORTANT GUIDELINES FOR CONTENT VERIFICATION:
- Use EXTRACT (not observe) when you need to verify text content, titles, or specific text values
- Use OBSERVE only for checking element existence or visual properties
- Extract is much more reliable for text-based verification

STEP SELECTION GUIDE:
1. Navigate to the specified page
2. For text content verification (titles, specific words, etc.): Use EXTRACT
3. For element existence verification: Use OBSERVE
4. Take a screenshot for evidence
5. Focus on specific, testable criteria

Example for text verification:
{
  "steps": [
    {
      "type": "navigate",
      "url": "https://example.com/page"
    },
    {
      "type": "extract",
      "instruction": "Extract the page title text"
    },
    {
      "type": "screenshot",
      "path": "screenshots/verification.png",
      "fullPage": false
    }
  ],
  "testCriteria": [
    "Page title contains the expected text"
  ]
}

Example for element existence:
{
  "steps": [
    {
      "type": "navigate",
      "url": "https://example.com/page"
    },
    {
      "type": "observe",
      "instruction": "Find the navigation menu on the page"
    },
    {
      "type": "screenshot",
      "path": "screenshots/verification.png",
      "fullPage": false
    }
  ],
  "testCriteria": [
    "Navigation menu exists on the page"
  ]
}`;

export class SimpleEvaluator {
  stagehand: any;
  model: any;
  currentProvider: "anthropic" | "openai";
  sessionId: string;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    // Determine which LLM provider to use based on environment
    this.currentProvider =
      process.env.LLM_PROVIDER === "anthropic" ? "anthropic" : "openai";

    // Initialize the AI model for generateObject based on provider
    this.model = this.createModel(this.currentProvider);
  }

  createModel(provider: "anthropic" | "openai") {
    if (provider === "anthropic") {
      return anthropic("claude-3-5-sonnet-20241022");
    } else {
      return openai("gpt-4o");
    }
  }

  async initStagehand() {
    if (!this.stagehand) {
      this.stagehand = await getOrCreateStagehand(this.sessionId);
    }
    return this.stagehand;
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private isRetryableError(error: any): boolean {
    return (
      error.message?.includes("529") ||
      error.message?.includes("Overloaded") ||
      error.message?.includes("rate limit") ||
      error.status === 429 ||
      error.status === 529
    );
  }

  private calculateBackoffDelay(attempt: number): number {
    // Exponential backoff: 1s, 2s, 4s with jitter
    const baseDelay = Math.min(1000 * Math.pow(2, attempt), 8000);
    const jitter = baseDelay * 0.3 * (Math.random() - 0.5); // ±30% jitter
    return Math.max(500, baseDelay + jitter); // minimum 500ms
  }

  async generateSteps(
    prompt: string,
    baseUrl: string
  ): Promise<{
    steps: Step[];
    testCriteria: string[];
  }> {
    const userMessage = `Please create Stagehand test steps for this prompt:\n\n${prompt}\n\nBase URL: ${baseUrl}`;

    console.log(
      `🤖 Using generateObject with ${this.currentProvider} provider...`
    );

    // Try current provider up to 3 times with exponential backoff
    let lastError: any;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        if (attempt > 0) {
          const delay = this.calculateBackoffDelay(attempt - 1);
          console.log(
            `⏳ Waiting ${Math.round(delay)}ms before retry ${attempt + 1}/3...`
          );
          await this.sleep(delay);
        }

        const result = await generateObject({
          model: this.model,
          system: SYSTEM_PROMPT,
          prompt: userMessage,
          schema: StepGenerationSchema,
          temperature: 0.1,
        });

        console.log(
          `✅ Successfully generated structured response with ${
            this.currentProvider
          }${attempt > 0 ? ` (attempt ${attempt + 1})` : ""}!`
        );
        console.log(
          "🔍 DEBUG: Generated test criteria:",
          result.object.testCriteria
        );
        console.log(
          "🔍 DEBUG: Generated steps:",
          JSON.stringify(result.object.steps, null, 2)
        );

        return {
          steps: result.object.steps as Step[],
          testCriteria: result.object.testCriteria,
        };
      } catch (error: any) {
        lastError = error;
        console.error(
          `❌ Attempt ${attempt + 1}/3 failed with ${this.currentProvider}:`,
          error.message
        );

        if (!this.isRetryableError(error)) {
          // Non-retryable error, fail immediately
          throw new Error(`Step generation failed: ${error}`);
        }
      }
    }

    // All retries with primary provider failed, try fallback provider
    const fallbackProvider =
      this.currentProvider === "anthropic" ? "openai" : "anthropic";
    console.log(
      `🔄 Primary provider exhausted, switching to ${fallbackProvider} provider...`
    );

    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        if (attempt > 0) {
          const delay = this.calculateBackoffDelay(attempt - 1);
          console.log(
            `⏳ Waiting ${Math.round(delay)}ms before fallback retry ${
              attempt + 1
            }/2...`
          );
          await this.sleep(delay);
        }

        const fallbackModel = this.createModel(fallbackProvider);
        const fallbackResult = await generateObject({
          model: fallbackModel,
          system: SYSTEM_PROMPT,
          prompt: userMessage,
          schema: StepGenerationSchema,
          temperature: 0.1,
        });

        console.log(
          `✅ Successfully generated steps with ${fallbackProvider} fallback${
            attempt > 0 ? ` (attempt ${attempt + 1})` : ""
          }!`
        );
        console.log(
          "🔍 DEBUG: Generated test criteria:",
          fallbackResult.object.testCriteria
        );
        console.log(
          "🔍 DEBUG: Generated steps:",
          JSON.stringify(fallbackResult.object.steps, null, 2)
        );

        return {
          steps: fallbackResult.object.steps as Step[],
          testCriteria: fallbackResult.object.testCriteria,
        };
      } catch (fallbackError: any) {
        console.error(
          `❌ Fallback attempt ${
            attempt + 1
          }/2 failed with ${fallbackProvider}:`,
          fallbackError.message
        );

        if (!this.isRetryableError(fallbackError)) {
          throw new Error(
            `Step generation failed with both providers. Primary: ${lastError.message}, Fallback: ${fallbackError.message}`
          );
        }
      }
    }

    throw new Error(
      `Step generation failed with both providers after all retries. Primary: ${lastError.message}`
    );
  }

  async executeSteps(steps: Step[], baseUrl: string): Promise<any[]> {
    await this.initStagehand();
    const results: any[] = [];
    console.log("🔍 DEBUG: Generated steps:\n", JSON.stringify(steps, null, 2));

    try {
      console.log(`🚀 Executing ${steps.length} steps...`);

      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        console.log(`\n📍 Step ${i + 1}/${steps.length}: ${step.type}`);

        try {
          switch (step.type) {
            case "navigate":
              await this.stagehand.page.goto(step.url);
              console.log(`✅ Navigated to ${step.url}`);
              break;

            case "observe":
              const observeResult = await this.stagehand.page.observe(
                step.instruction || ""
              );
              results[i] = observeResult;
              console.log(`✅ Observed: ${step.instruction}`);
              console.log(
                `🔍 DEBUG: Raw observe result:`,
                JSON.stringify(observeResult, null, 2)
              );
              console.log(
                `📊 Found ${
                  Array.isArray(observeResult)
                    ? observeResult.length
                    : "unknown"
                } results`
              );
              break;

            case "extract":
              try {
                const extractResult = await this.stagehand.page.extract(
                  step.instruction || "",
                  z.object({
                    content: z.string().describe("The extracted text content"),
                  })
                );
                results[i] = extractResult;
                console.log(`✅ Extracted: ${step.instruction}`);
                console.log(
                  `🔍 DEBUG: Raw extract result:`,
                  JSON.stringify(extractResult, null, 2)
                );
              } catch (extractError: any) {
                console.log(
                  `⚠️ Extract step failed, will retry with exponential backoff...`
                );

                if (this.isRetryableError(extractError)) {
                  // Try up to 2 more times with exponential backoff
                  let lastRetryError = extractError;
                  let success = false;

                  for (let retryAttempt = 0; retryAttempt < 2; retryAttempt++) {
                    const delay = this.calculateBackoffDelay(retryAttempt);
                    console.log(
                      `⏳ Waiting ${Math.round(delay)}ms before extract retry ${
                        retryAttempt + 1
                      }/2...`
                    );
                    await this.sleep(delay);

                    try {
                      const retryResult = await this.stagehand.page.extract(
                        step.instruction || "",
                        z.object({
                          content: z
                            .string()
                            .describe("The extracted text content"),
                        })
                      );
                      results[i] = retryResult;
                      console.log(
                        `✅ Extracted (retry ${retryAttempt + 1}): ${
                          step.instruction
                        }`
                      );
                      console.log(
                        `🔍 DEBUG: Retry extract result:`,
                        JSON.stringify(retryResult, null, 2)
                      );
                      success = true;
                      break;
                    } catch (retryError: any) {
                      lastRetryError = retryError;
                      console.log(
                        `❌ Extract retry ${retryAttempt + 1}/2 failed:`,
                        retryError.message
                      );
                    }
                  }

                  if (!success) {
                    console.log(
                      `❌ All extract retries exhausted. Provider overload persists.`
                    );
                    results[i] = {
                      error: `Extract failed after all retries: ${lastRetryError.message}`,
                    };
                  }
                } else {
                  console.log(
                    `❌ Extract failed with non-retryable error:`,
                    extractError
                  );
                  results[i] = { error: extractError.message };
                }
              }
              break;

            case "screenshot":
              await this.stagehand.page.screenshot({
                path: step.path,
                fullPage: step.fullPage ?? false,
              });
              console.log(`✅ Screenshot saved to ${step.path}`);
              break;

            default:
              console.log(`⚠️ Unknown step type: ${(step as any).type}`);
              break;
          }

          // Wait between steps
          if (i < steps.length - 1) {
            await this.stagehand.page.waitForTimeout(1000);
          }
        } catch (error) {
          console.error(`❌ Error in step ${i + 1}:`, error);
          results[i] = { error: (error as Error).message };
        }
      }

      console.log("\n✅ Steps execution complete!");
      return results;
    } finally {
      // Don't close the shared stagehand instance
      // await this.stagehand.close();
    }
  }

  async evaluateResults(
    stepResults: any[],
    steps: Step[],
    testCriteria: string[]
  ): Promise<TestResult> {
    const failedCriteria: string[] = [];

    console.log(
      `🔍 DEBUG: Starting evaluation with ${testCriteria.length} test criteria:`,
      testCriteria
    );
    console.log(
      `🔍 DEBUG: Step results:`,
      JSON.stringify(stepResults, null, 2)
    );

    // Analyze step results
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const result = stepResults[i];

      if (step.type === "observe") {
        console.log(`🔍 Evaluating observe step: ${step.instruction}`);
        console.log(`🔍 DEBUG: Full result object:`, result);
        console.log(
          `📊 Result type: ${
            Array.isArray(result)
              ? `array with ${result.length} items`
              : typeof result
          }`
        );

        // For observe steps, check if elements were found
        if (!result || result.error) {
          console.log(`❌ Step failed due to error: ${result?.error}`);
          console.log(
            `🔍 DEBUG: Adding to failed criteria: ${step.instruction}`
          );
          failedCriteria.push(step.instruction || "Unknown criterion");
        } else if (Array.isArray(result) && result.length === 0) {
          console.log(
            `❌ Step failed: No elements found matching the criteria`
          );
          console.log(
            `🔍 DEBUG: Adding to failed criteria: ${step.instruction}`
          );
          failedCriteria.push(step.instruction || "Unknown criterion");
        } else if (Array.isArray(result) && result.length > 0) {
          console.log(
            `✅ Step passed: Found ${result.length} matching elements`
          );
          console.log(`🔍 DEBUG: Step content:`, result);
        } else {
          console.log(`❌ Step failed: Unexpected result structure`);
          console.log(
            `🔍 DEBUG: Adding to failed criteria: ${step.instruction}`
          );
          failedCriteria.push(step.instruction || "Unknown criterion");
        }
      } else if (step.type === "extract") {
        console.log(`🔍 Evaluating extract step: ${step.instruction}`);
        console.log(`🔍 DEBUG: Extract result:`, result);

        // For extract steps, verify the content against test criteria
        const extractedContent = result?.extraction || result?.content || "";
        console.log(`🔍 DEBUG: Extracted content: "${extractedContent}"`);

        // Check if extracted content matches any of our test criteria
        let criteriaMatched = false;
        const contentLower = extractedContent.toLowerCase();

        for (const criterion of testCriteria) {
          console.log(`🔍 DEBUG: Checking criterion: "${criterion}"`);

          // Extract quoted words and other significant words from the criterion
          const quotedMatches = criterion.match(/['"]([^'"]+)['"]/g) || [];
          const wordMatches = criterion.match(/\b[a-zA-Z]{3,}\b/g) || [];

          const wordsToCheck = [
            ...quotedMatches.map((w) => w.replace(/['"]/g, "")),
            ...wordMatches,
          ].filter(
            (w) =>
              w.length > 2 &&
              ![
                "the",
                "and",
                "that",
                "contains",
                "word",
                "page",
                "title",
              ].includes(w.toLowerCase())
          );

          console.log(
            `🔍 DEBUG: Words to check: ${JSON.stringify(wordsToCheck)}`
          );

          for (const word of wordsToCheck) {
            const wordLower = word.toLowerCase();
            if (contentLower.includes(wordLower)) {
              console.log(
                `🔍 DEBUG: Found matching word: "${wordLower}" in content`
              );
              criteriaMatched = true;
              break;
            }
          }

          if (criteriaMatched) break;
        }

        if (!result || result.error) {
          console.log(`❌ Extract failed due to error: ${result?.error}`);
          failedCriteria.push(step.instruction || "Unknown criterion");
        } else if (!criteriaMatched) {
          console.log(`❌ Extract failed: Content doesn't match test criteria`);
          failedCriteria.push(step.instruction || "Unknown criterion");
        } else {
          console.log(`✅ Extract passed: Content matches criteria`);
        }
      }
    }

    const status = failedCriteria.length === 0 ? "PASS" : "FAIL";
    console.log(
      `🔍 DEBUG: Final evaluation - Failed criteria count: ${failedCriteria.length}`
    );
    console.log(`🔍 DEBUG: Failed criteria:`, failedCriteria);
    console.log(`🔍 DEBUG: Final status: ${status}`);

    const description =
      status === "PASS"
        ? "All criteria were successfully verified."
        : `Failed to verify the following criteria:\n${failedCriteria
            .map((c) => `- ${c}`)
            .join("\n")}`;

    return {
      status,
      description,
      failedCriteria: failedCriteria.length > 0 ? failedCriteria : undefined,
    };
  }

  async runTest(prompt: string, baseUrl: string): Promise<TestResult> {
    console.log("🔍 Generating test steps...");
    const { steps, testCriteria } = await this.generateSteps(prompt, baseUrl);

    console.log("📋 Generated test criteria:", testCriteria);
    console.log("🔧 Generated steps:", steps.length);

    console.log("🚀 Executing test steps...");
    const stepResults = await this.executeSteps(steps, baseUrl);

    console.log("📊 Evaluating results...");
    const result = await this.evaluateResults(stepResults, steps, testCriteria);

    return result;
  }
}
