// Shared helpers for the gateway scripts.
//
// Bifrost runs on http://localhost:8181 (mapped from container :8080).
// Governance API: /api/governance/*  ->  https://docs.getbifrost.ai/features/governance/virtual-keys

export const BIFROST_URL = process.env.BIFROST_URL ?? "http://localhost:8181";
export const MCP_URL = process.env.MCP_URL ?? "http://localhost:3355";

async function bf(method: string, path: string, body?: unknown): Promise<any> {
  const res = await fetch(`${BIFROST_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let parsed: any;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = text;
  }
  if (!res.ok) {
    throw new Error(
      `${method} ${path} -> ${res.status}: ${typeof parsed === "string" ? parsed : JSON.stringify(parsed)}`
    );
  }
  return parsed;
}

export interface VirtualKey {
  id: string;
  name: string;
  value: string; // sk-bf-...
  is_active: boolean;
  [k: string]: any;
}

export async function listVirtualKeys(): Promise<VirtualKey[]> {
  const out = await bf("GET", "/api/governance/virtual-keys");
  // Response shape varies between versions; normalize.
  return out?.virtual_keys ?? out?.data ?? out ?? [];
}

export interface CreateVKOpts {
  name: string;
  description?: string;
  /** Per-VK daily/monthly $ cap. */
  budget?: { max_limit: number; reset_duration: string };
  /** Per-VK rate limit. */
  rate_limit?: {
    token_max_limit?: number;
    token_reset_duration?: string;
    request_max_limit?: number;
    request_reset_duration?: string;
  };
  /** Restrict to specific providers/models. Omit for "any configured". */
  provider_configs?: Array<{
    provider: string;
    weight?: number;
    /** Per-provider key restriction. Use ["*"] for "all keys", [] for "deny all". */
    key_ids?: string[];
    /** Per-provider model restriction. Use ["*"] for "all models", [] for "deny all". */
    allowed_models?: string[];
  }>;
}

/**
 * Default VK provider config: allow all three providers we care about, all
 * models, all keys.
 *
 * Per the Bifrost schema (v1.5.0+ semantics):
 *   - empty arrays  => deny all  (deny-by-default)
 *   - ["*"]         => allow all
 *
 * So both `key_ids` and `allowed_models` need `["*"]` for an unrestricted VK.
 */
const DEFAULT_PROVIDER_CONFIGS: NonNullable<CreateVKOpts["provider_configs"]> = [
  {
    provider: "anthropic",
    weight: 1.0,
    key_ids: ["*"],
    allowed_models: ["*"],
  },
  {
    provider: "openai",
    weight: 1.0,
    key_ids: ["*"],
    allowed_models: ["*"],
  },
  {
    provider: "openrouter",
    weight: 1.0,
    key_ids: ["*"],
    allowed_models: ["*"],
  },
  {
    // Bifrost calls Google's public Gemini API "gemini" (Vertex AI is "vertex").
    // Note: MCP's provider.ts internally calls it "google" — that name is only
    // used in the AI SDK provider factory; the gateway-side identifier here is
    // what matters for Bifrost governance.
    provider: "gemini",
    weight: 1.0,
    key_ids: ["*"],
    allowed_models: ["*"],
  },
];

export async function createVirtualKey(opts: CreateVKOpts): Promise<VirtualKey> {
  const body = {
    is_active: true,
    ...opts,
    provider_configs: opts.provider_configs ?? DEFAULT_PROVIDER_CONFIGS,
  };
  const out = await bf("POST", "/api/governance/virtual-keys", body);
  // Response is sometimes { virtual_key: {...} } and sometimes the object itself.
  return out?.virtual_key ?? out;
}

export async function deleteVirtualKey(id: string): Promise<void> {
  await bf("DELETE", `/api/governance/virtual-keys/${id}`);
}

export async function ping(): Promise<void> {
  // Bifrost's UI/API root returns 200 even with no providers.
  const res = await fetch(`${BIFROST_URL}/api/config`);
  if (!res.ok) throw new Error(`Bifrost not reachable at ${BIFROST_URL}: ${res.status}`);
}
