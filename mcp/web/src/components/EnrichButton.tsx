import { useState, useRef, useEffect, useMemo } from "react";
import { Sparkles, X, AlertTriangle } from "lucide-react";
import { useIngestion } from "@/stores/useIngestion";
import { useSettings } from "@/stores/useSettings";
import { useServerConfig } from "../stores/useServerConfig";
import { useGraphData } from "@/stores/useGraphData";
import { toast } from "sonner";
import { resolveRepoUrl } from "@/lib/utils";
import { apiFetch } from "@/lib/api";

const API_BASE = import.meta.env.VITE_API_BASE || "";

function getEnrichSnapshot(progressData: any) {
  const latestUpdate =
    progressData?.updates && progressData.updates.length > 0
      ? progressData.updates[progressData.updates.length - 1]
      : progressData?.progress ?? progressData?.result ?? progressData ?? {};

  const totalTokens = latestUpdate?.total_tokens ?? progressData?.result?.total_tokens ?? {};

  return {
    processed: Number(latestUpdate?.processed ?? progressData?.result?.processed ?? 0),
    total_cost: Number(latestUpdate?.total_cost ?? progressData?.result?.total_cost ?? 0),
    current_batch_size: Number(latestUpdate?.current_batch_size ?? 0),
    total_tokens: {
      input: Number(totalTokens?.input ?? 0),
      output: Number(totalTokens?.output ?? 0),
    },
  };
}

export function EnrichButton() {
  const [open, setOpen] = useState(false);
  const [enriching, setEnriching] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const pollIntervalRef = useRef<number | null>(null);
  const activeReqIdRef = useRef<string | null>(null);

  const [costLimit, setCostLimit] = useState("0.50");
  const [batchSize, setBatchSize] = useState("50");

  const { repoUrl: storedRepoUrl } = useIngestion();
  const data = useGraphData((s) => s.data);
  const { apiKey, model } = useSettings();
  const {
    hasLLMKey,
    loaded: serverConfigLoaded,
    load: loadServerConfig,
  } = useServerConfig();
  const { reset: resetGraph } = useGraphData();

  const repoUrl = useMemo(
    () => resolveRepoUrl(data, storedRepoUrl),
    [data, storedRepoUrl],
  );

  useEffect(() => {
    void loadServerConfig();
  }, [loadServerConfig]);

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

  useEffect(() => {
    const savedReqId = localStorage.getItem("stakgraph_enrich_req_id");
    if (savedReqId && activeReqIdRef.current !== savedReqId) {
      setEnriching(true);
      startPolling(savedReqId, toast.loading("Resuming enrichment job...", { description: `Target: ${repoUrl}` }));
    }
    return () => {
      if (pollIntervalRef.current != null) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [repoUrl]);

  const startPolling = (reqId: string, initialToastId?: string | number) => {
    if (activeReqIdRef.current === reqId && pollIntervalRef.current != null) {
      return;
    }

    if (pollIntervalRef.current != null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    localStorage.setItem("stakgraph_enrich_req_id", reqId);
    activeReqIdRef.current = reqId;
    let toastId = initialToastId;

    const pollInterval = window.setInterval(async () => {
      try {
        const pollRes = await apiFetch(`${API_BASE}/progress?request_id=${reqId}`);
        if (!pollRes.ok) return;

        const progressData = await pollRes.json();
        const snapshot = getEnrichSnapshot(progressData);

        if (progressData.status === "failed" || progressData.status === "error") {
           clearInterval(pollInterval);
            pollIntervalRef.current = null;
            activeReqIdRef.current = null;
           setEnriching(false);
           toast.error("Enrichment Failed", {
             id: toastId,
             description: progressData.error || "An error occurred.",
           });
           localStorage.removeItem("stakgraph_enrich_req_id");
           return;
        }

        if (progressData.status === "done" || progressData.status === "completed") {
           clearInterval(pollInterval);
            pollIntervalRef.current = null;
            activeReqIdRef.current = null;
           setEnriching(false);
           
           const finalCost = snapshot.total_cost;
           const finalProcessed = snapshot.processed;
           const tokenSummary =
             snapshot.total_tokens.input > 0 || snapshot.total_tokens.output > 0
               ? ` · tokens in ${snapshot.total_tokens.input.toLocaleString()} / out ${snapshot.total_tokens.output.toLocaleString()}`
               : "";

           toast.success("Enrichment Complete", {
              id: toastId,
              description: `Processed ${finalProcessed} nodes for $${Number(finalCost).toFixed(4)}${tokenSummary}. Reloading graph...`,
           });
           localStorage.removeItem("stakgraph_enrich_req_id");

           setTimeout(() => {
               resetGraph();
           }, 1000);
           return;
        }

        // Still running - update the toast from the live `progress` payload
        const currentCost = snapshot.total_cost;
        const currentProcessed = snapshot.processed;
        const detailParts: string[] = [];

        if (snapshot.current_batch_size > 0) {
          detailParts.push(`Batch ${snapshot.current_batch_size}`);
        }
        if (snapshot.total_tokens.input > 0 || snapshot.total_tokens.output > 0) {
          detailParts.push(
            `Tokens in ${snapshot.total_tokens.input.toLocaleString()} / out ${snapshot.total_tokens.output.toLocaleString()}`,
          );
        }

        const description =
          detailParts.join(" • ") || "Generating descriptions and embeddings...";

        if (!toastId) {
           toastId = toast.loading(`Enriching... Processed: ${currentProcessed} | Cost: $${Number(currentCost).toFixed(4)}`, {
             description,
           });
        } else {
           toast.loading(`Enriching... Processed: ${currentProcessed} | Cost: $${Number(currentCost).toFixed(4)}`, {
              id: toastId,
              description,
           });
        }

      } catch (pollErr) {
        console.error("Poll error", pollErr);
      }
    }, 3000);

    pollIntervalRef.current = pollInterval;
  };

  const handleEnrich = async () => {
    if (!repoUrl) {
      toast.error("No repository selected", {
        description: "Add or select a repository before enrichment.",
      });
      return;
    }

    if (!apiKey && serverConfigLoaded && !hasLLMKey) {
      toast.error("Missing API Key", {
        description: "Add an API key in Settings or configure one on the server.",
      });
      return;
    }

    setEnriching(true);
    setOpen(false);

    let toastId: string | number | undefined;

    try {
      toastId = toast.loading("Starting enrichment job...", {
        description: `Target: ${repoUrl}`,
      });

      const body: Record<string, unknown> = {
        repo_url: repoUrl,
        cost_limit: parseFloat(costLimit),
        batch_size: parseInt(batchSize, 10),
        model: model,
      };
      if (apiKey) body.apiKey = apiKey;

      const res = await apiFetch(`${API_BASE}/repo/describe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => null);

      if (!res.ok || data?.error || !data?.request_id) {
        throw new Error(data?.error || "Failed to start enrichment");
      }

      startPolling(data.request_id, toastId);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error("Failed to enrich", {
        id: toastId,
        description: msg,
      });
      setEnriching(false);
    }
  };

  return (
    <div ref={ref} className="relative flex items-center">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={enriching || !repoUrl}
        className={`flex items-center gap-1.5 text-xs transition-colors rounded-md py-1.5 px-2 ${
          enriching || !repoUrl
            ? "text-muted-foreground opacity-70 cursor-not-allowed"
            : open 
              ? "text-foreground bg-accent"
              : "text-muted-foreground hover:text-foreground hover:bg-white/5"
        }`}
        title={repoUrl ? "Uses AI to describe nodes & embeddings" : "Add a repository to enable enrichment"}
      >
        <Sparkles className={`size-4 ${enriching ? "animate-pulse text-primary" : ""}`} />
        <span className="font-medium">Enrich</span>
      </button>

      {open && (
         <div className="absolute right-0 top-full mt-2 w-72 z-50 rounded-xl border border-border bg-background shadow-lg p-4 flex flex-col gap-4">
           <div className="flex items-center justify-between">
             <span className="text-sm font-medium">Node Enrichment</span>
             <button
               type="button"
               onClick={() => setOpen(false)}
               className="text-muted-foreground hover:text-foreground transition-colors"
             >
               <X className="size-3.5" />
             </button>
           </div>
           
           {!apiKey && serverConfigLoaded && !hasLLMKey ? (
               <div className="flex flex-col gap-3 rounded-lg border border-yellow-500/40 bg-yellow-500/10 p-3">
                 <div className="flex items-center gap-2">
                    <AlertTriangle className="size-4 text-yellow-500" />
                    <span className="text-sm font-medium text-yellow-500">API Key Required</span>
                 </div>
                 <p className="text-xs text-muted-foreground leading-snug">
                    No API key was detected in Settings or on the server. Add one in the top right Settings menu before you enrich your graph.
                 </p>
               </div>
           ) : (
             <>
               <p className="text-xs text-muted-foreground leading-snug">
                 This job invokes the AI model to read source code nodes and generate description summaries and vector embeddings.
               </p>
    
               <div className="flex flex-col gap-1.5">
                 <label className="text-xs text-muted-foreground">Budget Limit ($)</label>
                 <input
                   type="number"
                   step="0.01"
                   min="0.01"
                   value={costLimit}
                   onChange={(e) => setCostLimit(e.target.value)}
                   className="h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
                 />
                 <p className="text-[10px] text-muted-foreground/60">Job will stop automatically if this amount is surpassed.</p>
               </div>
    
               <div className="flex flex-col gap-1.5">
                 <label className="text-xs text-muted-foreground">Concurrency Batch Size</label>
                 <input
                   type="number"
                   step="10"
                   min="10"
                   max="1000"
                   value={batchSize}
                   onChange={(e) => setBatchSize(e.target.value)}
                   className="h-9 rounded-md border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring/50"
                 />
               </div>
    
               <button
                 onClick={handleEnrich}
                 className="flex items-center justify-center gap-2 h-9 rounded-md bg-primary text-primary-foreground text-sm font-medium transition-colors hover:bg-primary/90 mt-2 w-full"
               >
                 <Sparkles className="size-4" />
                 Start Enrichment
               </button>
             </>
           )}
         </div>
      )}
    </div>
  );
}
