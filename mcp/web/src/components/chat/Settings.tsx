import { Settings2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useSettings, MODEL_OPTIONS } from "@/stores/useSettings";
import { useServerConfig } from "../../stores/useServerConfig";

export function SettingsToggle() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`flex items-center justify-center w-7 h-7 rounded-md transition-colors ${
          open
            ? "text-foreground bg-accent"
            : "text-muted-foreground hover:text-foreground hover:bg-accent"
        }`}
        aria-label="Settings"
      >
        <Settings2 className="size-4" />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-72 z-50 rounded-xl border border-border bg-background shadow-lg p-4 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Agent Settings</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="size-3.5" />
            </button>
          </div>

          <SettingsFields />
        </div>
      )}
    </div>
  );
}

function SettingsFields() {
  const { model, apiKey, githubToken, setModel, setApiKey, setGithubToken } = useSettings();
  const {
    hasLLMKey,
    loaded: serverConfigLoaded,
    load: loadServerConfig,
  } = useServerConfig();

  useEffect(() => {
    void loadServerConfig();
  }, [loadServerConfig]);

  return (
    <>
      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-muted-foreground">Model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring/50"
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-muted-foreground">API Key</label>
        <input
          type="password"
          placeholder="sk-ant-api03-..."
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground/40"
        />
        <p className="text-xs text-muted-foreground/60">
          {serverConfigLoaded && hasLLMKey
            ? "Optional — the server already has an LLM key configured. Any key entered here stays in your browser only."
            : "Stored locally in your browser only."}
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs text-muted-foreground">GitHub Token</label>
        <input
          type="password"
          placeholder="ghp_..."
          value={githubToken}
          onChange={(e) => setGithubToken(e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground/40"
        />
        <p className="text-xs text-muted-foreground/60">Required to generate concepts from GitHub PRs.</p>
      </div>
    </>
  );
}
