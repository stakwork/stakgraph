import {
  gitsee_context,
  RepoContextMode,
  FeaturesContextResult,
  FirstPassContextResult,
  GitseeContextResult,
} from "./explore.js";

export type ExplorationResult =
  | FeaturesContextResult
  | FirstPassContextResult
  | string; // services

export async function explore(
  prompt: string | any[],
  repoPath: string,
  mode: RepoContextMode = "first_pass"
): Promise<ExplorationResult> {
  const startTime = Date.now();
  console.log(`🤖 Starting ${mode} exploration...`);

  try {
    // Get raw JSON string from get_context
    const { result: jsonString } = await gitsee_context(prompt, repoPath, mode);
    console.log(
      `📋 Raw exploration result:`,
      jsonString.substring(0, 200) + "..."
    );

    if (mode === "services") {
      return jsonString; // Return raw string for services mode (not JSON)
    }

    // Parse the JSON string
    let parsedResult: any;
    try {
      parsedResult = JSON.parse(jsonString);
    } catch (parseError) {
      console.warn("⚠️ Failed to parse JSON, treating as raw summary");
      // Fallback: create a structured result with the raw string as summary
      if (mode === "first_pass") {
        parsedResult = {
          summary: jsonString,
          key_files: [],
          infrastructure: [],
          dependencies: [],
          user_stories: [],
          pages: [],
        };
      } else {
        parsedResult = {
          summary: jsonString,
          key_files: [],
          features: [],
        };
      }
    }

    // Validate and ensure proper structure based on mode
    let result: ExplorationResult;

    if (mode === "first_pass") {
      result = {
        summary: parsedResult.summary || jsonString,
        key_files: parsedResult.key_files || [],
        infrastructure: parsedResult.infrastructure || [],
        dependencies: parsedResult.dependencies || [],
        user_stories: parsedResult.user_stories || [],
        pages: parsedResult.pages || [],
      } as FirstPassContextResult;
    } else {
      result = {
        summary: parsedResult.summary || jsonString,
        key_files: parsedResult.key_files || [],
        features: parsedResult.features || [],
      } as FeaturesContextResult;
    }

    const endTime = Date.now();
    const duration = endTime - startTime;
    console.log(`✅ ${mode} exploration completed in ${duration}ms`);
    console.log(
      `📊 Result: ${
        result.key_files.length
      } key files, summary: ${result.summary.substring(0, 100)}...`
    );

    return result;
  } catch (error) {
    console.error(`💥 Exploration failed:`, error);

    // Return error result with proper structure
    if (mode === "first_pass") {
      return {
        summary: `Exploration failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        key_files: [],
        infrastructure: [],
        dependencies: [],
        user_stories: [],
        pages: [],
      } as FirstPassContextResult;
    } else {
      return {
        summary: `Exploration failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        key_files: [],
        features: [],
      } as FeaturesContextResult;
    }
  }
}
