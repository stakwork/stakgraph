import { create } from "zustand";

export interface ModelOption {
  label: string;
  value: string;
  provider: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { label: "Claude Sonnet 4.6 (Anthropic)", value: "sonnet", provider: "anthropic" },
  { label: "Claude Opus 4.6 (Anthropic)", value: "opus", provider: "anthropic" },
  { label: "Claude Haiku 4.5 (Anthropic)", value: "haiku", provider: "anthropic" },
  { label: "Gemini 3 Pro (Google)", value: "gemini", provider: "google" },
  { label: "GPT-5 (OpenAI)", value: "gpt", provider: "openai" },
  { label: "Kimi K2.5 (OpenRouter)", value: "kimi", provider: "openrouter" },
];

interface SettingsState {
  model: string;
  apiKey: string;
  githubToken: string;
  setModel: (model: string) => void;
  setApiKey: (apiKey: string) => void;
  setGithubToken: (token: string) => void;
}

function loadStored(): { model: string; apiKey: string; githubToken: string } {
  try {
    const raw = localStorage.getItem("stakgraph_settings");
    if (raw) return { githubToken: "", ...JSON.parse(raw) };
  } catch {
    // ignore
  }
  return { model: "sonnet", apiKey: "", githubToken: "" };
}

function persist(model: string, apiKey: string, githubToken: string) {
  try {
    localStorage.setItem("stakgraph_settings", JSON.stringify({ model, apiKey, githubToken }));
  } catch {
    // ignore
  }
}

export const useSettings = create<SettingsState>((set, get) => {
  const stored = loadStored();
  return {
    model: stored.model,
    apiKey: stored.apiKey,
    githubToken: stored.githubToken,
    setModel: (model) => {
      set({ model });
      persist(model, get().apiKey, get().githubToken);
    },
    setGithubToken: (githubToken) => {
      set({ githubToken });
      persist(get().model, get().apiKey, githubToken);
    },
    setApiKey: (apiKey) => {
      set({ apiKey });
      persist(get().model, apiKey, get().githubToken);
    },
  };
});
