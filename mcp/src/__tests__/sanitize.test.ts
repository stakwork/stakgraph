import { sanitizePrompt, sanitizeMessages } from "../utils/sanitize.js";

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(`FAIL: ${message}`);
  }
  console.log(`✓ ${message}`);
}

function assertEqual(actual: any, expected: any, message: string) {
  if (actual !== expected) {
    throw new Error(
      `FAIL: ${message}\n  Expected: ${expected}\n  Actual: ${actual}`
    );
  }
  console.log(`✓ ${message}`);
}

console.log("\n=== Testing sanitizePrompt ===\n");

assertEqual(
  sanitizePrompt("https://$domain/path"),
  "https://\\$domain/path",
  "escapes $ shell variables"
);

assertEqual(
  sanitizePrompt("$HOME and $USER"),
  "\\$HOME and \\$USER",
  "escapes multiple $ characters"
);

assertEqual(
  sanitizePrompt("hello\x00world"),
  "helloworld",
  "removes null bytes"
);
assertEqual(sanitizePrompt("hello\nworld"), "hello\nworld", "keeps newlines");
assertEqual(sanitizePrompt("hello\tworld"), "hello\tworld", "keeps tabs");

const excessiveSpaces = "hello" + " ".repeat(15) + "world";
assertEqual(
  sanitizePrompt(excessiveSpaces),
  "hello" + " ".repeat(5) + "world",
  "normalizes excessive whitespace"
);

const normalPrompt = "How does authentication work in this repo?";
assertEqual(
  sanitizePrompt(normalPrompt),
  normalPrompt,
  "preserves normal prompts unchanged"
);

const badPrompt = "<title> Bug </title> cal.com https://$domain/w/cal.com";
const result = sanitizePrompt(badPrompt);
assert(result.includes("\\$domain"), "escapes $domain in the failing case");

assertEqual(sanitizePrompt(""), "", "handles empty string");
assertEqual(sanitizePrompt(null), "", "handles null");
assertEqual(sanitizePrompt(undefined), "", "handles undefined");

assertEqual(sanitizePrompt(123 as any), "", "handles number input");
assertEqual(sanitizePrompt({} as any), "", "handles object input");

console.log("\n=== Testing sanitizeMessages ===\n");

const messages1 = [{ role: "user" as const, content: "Check $HOME directory" }];
const result1 = sanitizeMessages(messages1);
assertEqual(
  result1[0].content,
  "Check \\$HOME directory",
  "sanitizes string content in messages"
);

const messages2 = [
  {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Check $PATH" }],
  },
];
const result2 = sanitizeMessages(messages2);
assertEqual(
  (result2[0].content as any)[0].text,
  "Check \\$PATH",
  "sanitizes text parts in array content"
);
