import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";
import { toast } from "sonner";

const STANDALONE_BASE = import.meta.env.VITE_STANDALONE_URL || "http://localhost:7799";

export function SyncButton() {
  const [syncing, setSyncing] = useState(false);
  const { repoUrl, username, pat } = useIngestion();
  const { reset: resetGraph } = useGraphData();

  // If there's no repo loaded, we can't sync
  if (!repoUrl) return null;

  const handleSync = async () => {
    // If multiple repos are comma-separated, standalone sync_async currently only supports 1
    const urls = repoUrl.split(",").map((s) => s.trim()).filter(Boolean);
    if (urls.length > 1) {
      toast.error("Sync is only supported for a single repository.", {
        description: "You have multiple repositories ingested.",
      });
      return;
    }

    setSyncing(true);
    let syncToastId: string | number | undefined;
    
    try {
      syncToastId = toast.loading("Syncing repository...", {
        description: repoUrl,
      });

      const body: Record<string, string | boolean> = {
        repo_url: repoUrl,
      };

      if (username) body.username = username;
      if (pat) body.pat = pat;

      const res = await fetch(`${STANDALONE_BASE}/sync_async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => null);

      if (!res.ok || data?.error || !data?.request_id) {
        throw new Error(data?.error || "Failed to start sync");
      }

      // Sync request accepted! Start SSE listener purely for tracking
      const reqId = data.request_id;
      const eventSource = new EventSource(
        `${STANDALONE_BASE}/progress/${reqId}`
      );

      eventSource.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          const isComplete = update.status?.toLowerCase() === "complete";
          const isError =
            update.status?.toLowerCase() === "error" ||
            update.status?.toLowerCase() === "failed";
            
          if (update.message && !isComplete && !isError) {
             toast.loading(`Syncing: ${update.message}`, { id: syncToastId });
          }

          if (isComplete) {
            toast.success("Sync Complete", {
              id: syncToastId,
              description: "Reloading graph with latest data...",
            });
            eventSource.close();
            setSyncing(false);
            
            // Reload graph by resetting the store, forcing a refetch on next render
            setTimeout(() => {
               resetGraph();
            }, 500);
          } else if (isError) {
            toast.error("Sync Error", {
              id: syncToastId,
              description: update.message || "An error occurred during sync",
            });
            eventSource.close();
            setSyncing(false);
          }
        } catch (err) {
          console.error("Failed to parse sync update", err);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        // Don't show an error toast here immediately on disconnect, just reset state
        // because standard connection close can also trigger onerror
        setSyncing(false);
      };

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error("Failed to start sync", {
        id: syncToastId,
        description: msg,
      });
      setSyncing(false);
    }
  };

  return (
    <button
      onClick={handleSync}
      disabled={syncing}
      className={`flex items-center gap-1.5 text-xs transition-colors rounded-md py-1.5 px-2 ${
        syncing
          ? "text-muted-foreground opacity-70 cursor-not-allowed"
          : "text-muted-foreground hover:text-foreground hover:bg-white/5"
      }`}
      title="Sync Repository"
    >
      <RefreshCw className={`size-4 ${syncing ? "animate-spin text-primary" : ""}`} />
      <span className="font-medium">Sync</span>
    </button>
  );
}
