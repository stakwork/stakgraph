import { AiUsageWithLegacy } from "../aieo/src/usage.js";
import type { SessionReflection } from "../repo/session.js";

export type Json = Record<string, unknown> | undefined;

export interface Tool {
  name: string;
  description: string;
  inputSchema: Json;
}

export interface ContextResult {
  final: string;
  usage: AiUsageWithLegacy & {
    model?: string;
    provider?: string;
  };
  tool_use?: string;
  content: any;
  logs?: string;
  sessionId?: string; // Return session ID for multi-turn conversations
  // Concepts this session read, with the agent's ranking when `reflect` was
  // set. Cumulative over the session, and identical to what the reflection
  // sidecar holds — returned here so a caller that waited for the reflect
  // call doesn't need a second round-trip to GET /repo/agent/session.
  reflection?: SessionReflection;
}
