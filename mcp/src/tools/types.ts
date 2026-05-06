import type { AiUsage } from "../aieo/src/index.js";

export type Json = Record<string, unknown> | undefined;

export interface Tool {
  name: string;
  description: string;
  inputSchema: Json;
}

export interface ContextResult {
  final: string;
  usage: AiUsage & { model?: string; provider?: string };
  tool_use?: string;
  content: any;
  logs?: string;
  sessionId?: string; // Return session ID for multi-turn conversations
}
