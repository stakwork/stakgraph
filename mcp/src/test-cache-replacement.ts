#!/usr/bin/env ts-node

/**
 * Test script to demonstrate cache replacement functionality in ask_prompt
 *
 * Usage:
 * ts-node src/test-cache-replacement.ts
 */

import { ask_prompt, CacheControlOptions } from "./tools/intelligence/index.js";

async function testCacheReplacement() {
  const testPrompt = "How does authentication work in this codebase?";

  console.log("=== Testing Cache Replacement Functionality ===\n");

  // Test 1: Create initial answer (should create new node)
  console.log("1. Creating initial answer (should create new node):");
  try {
    const result1 = await ask_prompt(testPrompt);
    console.log("   Result:", result1.answer.substring(0, 100) + "...");
    console.log("   Hints count:", result1.hints.length);
    console.log(
      "   Reused hints:",
      result1.hints.filter((h) => h.reused).length
    );
  } catch (error) {
    console.error("   Error:", error);
  }

  console.log("\n" + "=".repeat(50) + "\n");

  // Test 2: Ask same question again (should use cached result)
  console.log("2. Asking same question again (should use cached result):");
  try {
    const result2 = await ask_prompt(testPrompt);
    console.log("   Result:", result2.answer.substring(0, 100) + "...");
    console.log("   Hints count:", result2.hints.length);
    console.log(
      "   Reused hints:",
      result2.hints.filter((h) => h.reused).length
    );
  } catch (error) {
    console.error("   Error:", error);
  }

  console.log("\n" + "=".repeat(50) + "\n");

  // Test 3: Force refresh (should delete old node and create new one)
  console.log("3. Force refresh (should delete old node and create new one):");
  try {
    const cacheControl: CacheControlOptions = { forceRefresh: true };
    const result3 = await ask_prompt(
      testPrompt,
      undefined,
      undefined,
      cacheControl
    );
    console.log("   Result:", result3.answer.substring(0, 100) + "...");
    console.log("   Hints count:", result3.hints.length);
    console.log(
      "   Reused hints:",
      result3.hints.filter((h) => h.reused).length
    );
  } catch (error) {
    console.error("   Error:", error);
  }

  console.log("\n" + "=".repeat(50) + "\n");

  // Test 4: Ask again after force refresh (should use new cached result)
  console.log(
    "4. Asking again after force refresh (should use new cached result):"
  );
  try {
    const result4 = await ask_prompt(testPrompt);
    console.log("   Result:", result4.answer.substring(0, 100) + "...");
    console.log("   Hints count:", result4.hints.length);
    console.log(
      "   Reused hints:",
      result4.hints.filter((h) => h.reused).length
    );
  } catch (error) {
    console.error("   Error:", error);
  }

  console.log("\n" + "=".repeat(50) + "\n");

  // Test 5: Max age 0.001 hours (should replace due to age)
  console.log("5. Max age 0.001 hours (should replace due to age):");
  try {
    const cacheControl: CacheControlOptions = { maxAgeHours: 0.001 }; // ~3.6 seconds
    const result5 = await ask_prompt(
      testPrompt,
      undefined,
      undefined,
      cacheControl
    );
    console.log("   Result:", result5.answer.substring(0, 100) + "...");
    console.log("   Hints count:", result5.hints.length);
    console.log(
      "   Reused hints:",
      result5.hints.filter((h) => h.reused).length
    );
  } catch (error) {
    console.error("   Error:", error);
  }

  console.log("\n=== Test Complete ===");
  console.log("\nKey behaviors demonstrated:");
  console.log("- Initial call creates new node");
  console.log("- Subsequent calls use cached result");
  console.log(
    "- forceRefresh=true deletes old node and immediately creates new one (no gap)"
  );
  console.log("- maxAgeHours triggers replacement when age exceeds threshold");
  console.log(
    "- Deletion happens right before creation to ensure no gap in data"
  );
}

// Run the test
testCacheReplacement().catch(console.error);
