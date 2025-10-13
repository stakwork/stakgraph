export type Json = Record<string, unknown> | undefined;

export interface Tool {
  name: string;
  description: string;
  inputSchema: Json;
}

export interface ContextResult {
  final: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
}
