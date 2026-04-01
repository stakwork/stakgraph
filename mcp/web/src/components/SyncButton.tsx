import { useState, useEffect, useRef, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";
import { toast } from "sonner";
import { resolveRepoUrl } from "@/lib/utils";

const STANDALONE_BASE = import.meta.env.VITE_STANDALONE_URL || "http://localhost:7799";

export function SyncButton() {
  const [syncing, setSyncing] = useState(false);
  const { repoUrl: storedRepoUrl, username, pat } = useIngestion();
  const data = useGraphData((s) => s.data);
  const { reset: resetGraph } = useGraphData();
  const pollIntervalRef = useRef<number | null>(null);
  const activeReqIdRef = useRef<string | null>(null);

  const repoUrl = useMemo(
    () => resolveRepoUrl(data, storedRepoUrl),
    [data, storedRepoUrl],
  );

  useEffect(() => {
    const savedReqId = localStorage.getItem("stakgraph_sync_req_id");
    if (savedReqId && activeReqIdRef.current !== savedReqId) {
      setSyncing(true);
      startListening(savedReqId, toast.loading("Resuming sync...", { description: repoUrl || "Sync in progress" }));
    }
    return () => {
      if (pollIntervalRef.current != null) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [repoUrl]);

  const startListening = (reqId: string, initialToastId?: string | number) => {
    if (activeReqIdRef.current === reqId && pollIntervalRef.current != null) {
      return;
    }

    if (pollIntervalRef.current != null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    localStorage.setItem("stakgraph_sync_req_id", reqId);
    activeReqIdRef.current = reqId;
    let syncToastId = initialToastId;

    const pollInterval = window.setInterval(async () => {
      try {
        const res = await fetch(`${STANDALONE_BASE}/status/${reqId}`);
        if (!res.ok) {
          // If 404 or backend is down, keep trying unless it's gone for a very long time
          // but we can just let it continue polling
          return;
        }

        const data = await res.json();
        const status = data.status; // "InProgress", "Complete", "Failed"
        const update = data.update; // The StatusUpdate struct

        if (status === "Complete") {
          toast.success("Sync Complete", {
            id: syncToastId,
            description: "Reloading graph with latest data...",
          });
          clearInterval(pollInterval);
          pollIntervalRef.current = null;
          activeReqIdRef.current = null;
          setSyncing(false);
          localStorage.removeItem("stakgraph_sync_req_id");
          
          setTimeout(() => resetGraph(), 500);
          return;
        } else if (status === "Failed") {
          toast.error("Sync Error", {
            id: syncToastId,
            description: update?.message || "An error occurred during sync",
          });
          clearInterval(pollInterval);
          pollIntervalRef.current = null;
          activeReqIdRef.current = null;
          setSyncing(false);
          localStorage.removeItem("stakgraph_sync_req_id");
          return;
        }

        // InProgress
        if (update && update.message) {
           if (!syncToastId) {
             syncToastId = toast.loading(`Syncing: ${update.message}`);
           } else {
             toast.loading(`Syncing: ${update.message}`, { id: syncToastId });
           }
        }
      } catch (err) {
        console.error("Failed to poll sync status", err);
      }
    }, 2000);

    pollIntervalRef.current = pollInterval;
  };

  const handleSync = async () => {
    if (!repoUrl) {
      toast.error("No repository selected", {
        description: "Add or select a repository before syncing.",
      });
      return;
    }
    const urls = repoUrl.split(",").map((s) => s.trim()).filter(Boolean);
    if (urls.length > 1) {
      toast.error("Sync is only supported for a single repository.", {
        description: "You have multiple repositories ingested.",
      });
      return;
    }

    setSyncing(true);
    let syncToastId = toast.loading("Starting sync...", {
      description: repoUrl,
    });
    
    try {
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

      startListening(data.request_id, syncToastId);
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
      disabled={syncing || !repoUrl}
      className={`flex items-center gap-1.5 text-xs transition-colors rounded-md py-1.5 px-2 ${
        syncing || !repoUrl
          ? "text-muted-foreground opacity-70 cursor-not-allowed"
          : "text-muted-foreground hover:text-foreground hover:bg-white/5"
      }`}
      title={repoUrl ? "Sync Repository" : "Add a repository to enable sync"}
    >
      <RefreshCw className={`size-4 ${syncing ? "animate-spin text-primary" : ""}`} />
      <span className="font-medium">Sync</span>
    </button>
  );
}
