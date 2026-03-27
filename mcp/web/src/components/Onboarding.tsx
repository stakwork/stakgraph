import { useState } from "react";
import {
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIngestion, type RepoEntry } from "@/stores/useIngestion";
import { cleanError } from "@/lib/errors";

const STANDALONE_BASE = import.meta.env.VITE_STANDALONE_URL || "http://localhost:7799";

const emptyRepo = (): RepoEntry => ({ url: "", username: "", pat: "" });

interface OnboardingProps {
  onStarted: () => void;
}

export function Onboarding({ onStarted }: OnboardingProps) {
  const [repos, setRepos] = useState<RepoEntry[]>([emptyRepo()]);
  const [useLsp, setUseLsp] = useState(false);
  const [branch, setBranch] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [privateOpen, setPrivateOpen] = useState<Record<number, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { setRunning, setRepo } = useIngestion();

  const updateRepo = (i: number, field: keyof RepoEntry, value: string) => {
    setRepos((prev) => prev.map((r, idx) => (idx === i ? { ...r, [field]: value } : r)));
  };

  const addRepo = () => setRepos((prev) => [...prev, emptyRepo()]);
  const removeRepo = (i: number) => setRepos((prev) => prev.filter((_, idx) => idx !== i));

  const togglePrivate = (i: number) =>
    setPrivateOpen((prev) => ({ ...prev, [i]: !prev[i] }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);

    const urls = repos.map((r) => r.url.trim()).filter(Boolean);
    if (urls.length === 0) return;

    // Use the first repo's credentials (multi-repo with same PAT is the common case)
    const firstWithCreds = repos.find((r) => r.username || r.pat);

    const body: Record<string, unknown> = {
      repo_url: urls.join(","),
      use_lsp: useLsp,
      realtime: true,
    };

    if (branch.trim()) body.branch = branch.trim();
    if (firstWithCreds?.username) body.username = firstWithCreds.username;
    if (firstWithCreds?.pat) body.pat = firstWithCreds.pat;

    setSubmitting(true);
    try {
      const res = await fetch(`${STANDALONE_BASE}/ingest_async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok || data?.error || !data?.request_id) {
        throw new Error(data?.error || `${res.status} ${res.statusText}`);
      }
      setRunning(data.request_id);
      setRepo(
        urls.join(","),
        firstWithCreds?.username || undefined,
        firstWithCreds?.pat || undefined,
      );
      onStarted();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSubmitError(cleanError(msg));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-full w-full px-4">
      <div className="w-full max-w-lg">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold tracking-tight mb-1">
            Add a repository
          </h1>
          <p className="text-sm text-muted-foreground">
            Stakgraph will build your local knowledge graph.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {repos.map((repo, i) => (
            <div
              key={i}
              className="flex flex-col gap-2 rounded-lg border border-border p-4"
            >
              <div className="flex items-center gap-2">
                <input
                  type="url"
                  placeholder="https://github.com/org/repo.git"
                  value={repo.url}
                  onChange={(e) => updateRepo(i, "url", e.target.value)}
                  required
                  className="flex-1 h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground"
                />
                {repos.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeRepo(i)}
                    className="text-muted-foreground hover:text-destructive transition-colors"
                  >
                    <Trash2 className="size-4" />
                  </button>
                )}
              </div>

              <button
                type="button"
                onClick={() => togglePrivate(i)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-fit"
              >
                {privateOpen[i] ? (
                  <ChevronUp className="size-3" />
                ) : (
                  <ChevronDown className="size-3" />
                )}
                Private repo credentials
              </button>

              {privateOpen[i] && (
                <div className="flex gap-2 mt-1">
                  <input
                    type="text"
                    placeholder="Username"
                    value={repo.username}
                    onChange={(e) => updateRepo(i, "username", e.target.value)}
                    className="flex-1 h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground"
                  />
                  <input
                    type="password"
                    placeholder="Personal Access Token"
                    value={repo.pat}
                    onChange={(e) => updateRepo(i, "pat", e.target.value)}
                    className="flex-1 h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground"
                  />
                </div>
              )}
            </div>
          ))}

          <button
            type="button"
            onClick={addRepo}
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors w-fit"
          >
            <Plus className="size-4" />
            Add another repo
          </button>

          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-fit"
          >
            {showAdvanced ? (
              <ChevronUp className="size-3" />
            ) : (
              <ChevronDown className="size-3" />
            )}
            Advanced options
          </button>

          {showAdvanced && (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-4">
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={useLsp}
                    onChange={(e) => setUseLsp(e.target.checked)}
                    className="rounded border-input accent-primary"
                  />
                  Use LSP
                  <span className="text-xs text-muted-foreground">
                    (slower, more accurate cross-file links)
                  </span>
                </label>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Branch (optional)"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50 placeholder:text-muted-foreground w-52"
                />
              </div>
            </div>
          )}

          {submitError && (
            <div className="flex items-start gap-2.5 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2.5">
              <AlertCircle className="size-4 text-destructive shrink-0 mt-0.5" />
              <div className="flex flex-col gap-0.5">
                <p className="text-sm text-destructive leading-snug">
                  {submitError}
                </p>
                {/auth|credential|token|password|forbidden|401|403/i.test(
                  submitError,
                ) && (
                  <p className="text-xs text-muted-foreground">
                    Check your username and Personal Access Token under "Private
                    repo credentials".
                  </p>
                )}
              </div>
            </div>
          )}

          <Button type="submit" disabled={submitting} className="mt-2">
            {submitting ? "Starting…" : "Start ingestion"}
          </Button>
        </form>
      </div>
    </div>
  );
}


