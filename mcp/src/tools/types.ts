export type Json = Record<string, unknown> | undefined;

export interface Tool {
  name: string;
  description: string;
  inputSchema: Json;
}
