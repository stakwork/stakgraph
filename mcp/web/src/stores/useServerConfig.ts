import { create } from "zustand";
import { apiFetch } from "@/lib/api";

const API_BASE = import.meta.env.VITE_API_BASE || "";

type ProviderAvailability = Record<string, boolean>;

interface ServerConfigState {
  providers: ProviderAvailability;
  hasLLMKey: boolean;
  defaultProvider: string | null;
  loading: boolean;
  loaded: boolean;
  load: () => Promise<void>;
}

export const useServerConfig = create<ServerConfigState>((set, get) => ({
  providers: {},
  hasLLMKey: false,
  defaultProvider: null,
  loading: false,
  loaded: false,
  load: async () => {
    if (get().loading || get().loaded) return;

    set({ loading: true });

    try {
      const res = await apiFetch(`${API_BASE}/server-config`);
      const data = await res.json().catch(() => ({}));
      const providers =
        data && typeof data.providers === "object" && data.providers
          ? (data.providers as ProviderAvailability)
          : {};

      set({
        providers,
        hasLLMKey:
          Boolean(data?.has_llm_key) || Object.values(providers).some(Boolean),
        defaultProvider:
          typeof data?.default_provider === "string"
            ? data.default_provider
            : null,
        loading: false,
        loaded: true,
      });
    } catch {
      set({
        providers: {},
        hasLLMKey: false,
        defaultProvider: null,
        loading: false,
        loaded: true,
      });
    }
  },
}));
