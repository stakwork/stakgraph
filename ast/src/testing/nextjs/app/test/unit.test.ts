// @ts-nocheck
import { cn } from "../../lib/utils";

// Helper functions - should be classified as Functions, not Tests
function formatTestOutput(label: string, value: any): string {
  return `${label}: ${JSON.stringify(value)}`;
}

async function waitForCondition(condition: () => boolean, timeout: number = 1000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (condition()) return true;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  return false;
}

describe("unit: utils.cn", () => {
  it("merges class names", () => {
    const result = cn("btn", "btn-primary");
    expect(result).toBe("btn btn-primary");
    console.log("cn result:", result);
  });
});

describe("unit: types exist", () => {
  it("has type definitions for Button variants", () => {
    const { buttonVariants } = require("../../components/ui/button");
    console.log("buttonVariants keys:", Object.keys(buttonVariants || {}));
  });
});
