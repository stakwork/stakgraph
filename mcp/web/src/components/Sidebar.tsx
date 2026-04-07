import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { ChevronDown, BookOpen, Lightbulb, Zap, Loader2, Activity } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/useApi";
import { useGraphData } from "@/stores/useGraphData";
import { toast } from "sonner";
import { useIngestion } from "@/stores/useIngestion";
import { useSettings } from "@/stores/useSettings";
import { ImportanceLens } from "@/components/ImportanceLens";
import type { Doc, FeaturesResponse, FeatureSummary } from "@/types";

// GET /docs returns: [ { "repo-name": { documentation: "..." } }, ... ]
type DocsResponse = Array<Record<string, { documentation: string }>>;
type DocsStatus = {
  type: "success" | "empty" | "error";
  message: string;
} | null;

function parseDocs(raw: DocsResponse | null): Doc[] {
  if (!raw) return [];
  return raw.map((entry) => {
    const repoName = Object.keys(entry)[0];
    return { repoName, documentation: entry[repoName].documentation };
  });
}

function getRepoFromFeatureId(id: string): string {
  const parts = id.split("/");
  return parts.slice(0, 2).join("/");
}

interface SidebarProps {
  activeItemKey: string | null;
  onDocClick: (repoName: string, documentation: string) => void;
  onConceptClick: (id: string, name: string, description: string) => void;
}

export function Sidebar({
  activeItemKey,
  onDocClick,
  onConceptClick,
}: SidebarProps) {
  const setHighlightedFeature = useGraphData((s) => s.setHighlightedFeature);
  const ingestedRepoUrl = useIngestion((s) => s.repoUrl);
  const githubToken = useSettings((s) => s.githubToken);
  const { model: settingsModel, apiKey: settingsApiKey } = useSettings();
  const clearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleConceptHover = useCallback(
    (refId: string | null) => {
      if (clearTimer.current) {
        clearTimeout(clearTimer.current);
        clearTimer.current = null;
      }
      if (refId) {
        setHighlightedFeature(refId);
      } else {
        // Debounce the clear so moving between items doesn't flicker
        clearTimer.current = setTimeout(() => {
          setHighlightedFeature(null);
        }, 100);
      }
    },
    [setHighlightedFeature]
  );

  const {
    data: rawDocs,
    loading: isDocsLoading,
    refetch: refetchDocs,
  } = useApi<DocsResponse>("/docs");
  const {
    data: featuresData,
    loading: isConceptsLoading,
    refetch: refetchConcepts,
  } = useApi<FeaturesResponse>("/gitree/features");

  const API_BASE = import.meta.env.VITE_API_BASE || "";

  // ── Docs: synchronous POST, stays disabled until it resolves ──────────
  const [isGeneratingDocs, setIsGeneratingDocs] = useState(false);
  const [docsStatus, setDocsStatus] = useState<DocsStatus>(null);

  const handleGenerateDocs = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isGeneratingDocs) return;

      setIsGeneratingDocs(true);
      setDocsStatus(null);

      const toastId = toast.loading("Generating docs...", {
        description: "Scanning repository rules files and guidance.",
      });

      try {
        const body: Record<string, unknown> = {};
        if (settingsModel) body.model = settingsModel;
        if (settingsApiKey) body.apiKey = settingsApiKey;

        const res = await fetch(`${API_BASE}/learn_docs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        const payload = await res.json().catch(() => null);

        if (!res.ok) {
          throw new Error(payload?.error || `Server error (HTTP ${res.status})`);
        }

        refetchDocs();

        const createdCount =
          payload?.summaries && typeof payload.summaries === "object"
            ? Object.keys(payload.summaries).length
            : 0;

        if (createdCount > 0) {
          const message =
            createdCount === 1
              ? "Documentation created for 1 repository."
              : `Documentation created for ${createdCount} repositories.`;
          setDocsStatus({ type: "success", message });
          toast.dismiss(toastId);
          toast.success("Docs generated", {
            description: message,
            duration: 4000,
          });
        } else {
          const message =
            "No new documentation was generated. Docs may already exist, or no rules files were found.";
          setDocsStatus({ type: "empty", message });
          toast.dismiss(toastId);
          toast("No new docs generated", {
            description: message,
            duration: 5000,
          });
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to generate docs.";
        setDocsStatus({ type: "error", message });
        toast.dismiss(toastId);
        toast.error("Docs generation failed", {
          description: message,
          duration: 6000,
        });
      } finally {
        setIsGeneratingDocs(false);
      }
    },
    [API_BASE, isGeneratingDocs, refetchDocs, settingsModel, settingsApiKey],
  );

  // ── Concepts: async POST + poll processing flag every 3 s ─────────────
  const [conceptsTriggered, setConceptsTriggered] = useState(false);
  const [conceptsError, setConceptsError] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollCount = useRef(0);

  const serverProcessing = featuresData?.processing ?? false;
  const isGeneratingConcepts = conceptsTriggered || serverProcessing;

  // Start polling as soon as triggered or server says processing
  useEffect(() => {
    console.log("[concepts] poll effect", { isGeneratingConcepts, conceptsTriggered, serverProcessing });
    if (isGeneratingConcepts) {
      if (!pollTimer.current) {
        pollCount.current = 0;
        pollTimer.current = setInterval(() => {
          pollCount.current += 1;
          console.log(`[concepts] poll #${pollCount.current}`, { conceptsTriggered, serverProcessing });
          if (
            conceptsTriggered &&
            !serverProcessing &&
            pollCount.current >= 10
          ) {
            console.warn("[concepts] timed out waiting for server processing flag");
            setConceptsError(
              "Generation may have failed — no server response. Check logs and try again.",
            );
            setConceptsTriggered(false);
            return;
          }
          refetchConcepts();
        }, 3000);
      }
    } else {
      if (pollTimer.current) {
        console.log("[concepts] stopping poll timer");
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }

      setConceptsTriggered(false);
    }
    return () => {
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [
    isGeneratingConcepts,
    conceptsTriggered,
    serverProcessing,
    refetchConcepts,
  ]);

  const handleGenerateConcepts = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isGeneratingConcepts) return;
      setConceptsError(null);

      if (!githubToken) {
        setConceptsError("A GitHub token is required. Add it in Settings (⚙).");
        return;
      }

      console.log("[concepts] ingestedRepoUrl:", ingestedRepoUrl);

      if (!ingestedRepoUrl) {
        setConceptsError("No repository detected. Load a graph first.");
        return;
      }

      const firstUrl = ingestedRepoUrl.split(",")[0].trim();
      console.log("[concepts] firing POST /gitree/process with repo_url:", firstUrl);

      setConceptsTriggered(true);

      const qp = new URLSearchParams();
      qp.set("repo_url", firstUrl);
      qp.set("token", githubToken);
      const url = `${API_BASE}/gitree/process?${qp.toString()}`;

      fetch(url, { method: "POST" })
        .then(async (r) => {
          console.log("[concepts] POST /gitree/process response:", r.status, r.ok);
          if (!r.ok) {
            const body = await r.json().catch(() => ({}));
            const msg =
              (body as any)?.error || `Server error (HTTP ${r.status})`;
            console.error("[concepts] server rejected request:", msg, body);
            setConceptsError(msg);
            setConceptsTriggered(false);
          } else {
            const body = await r.json().catch(() => ({}));
            console.log("[concepts] accepted, request_id:", (body as any)?.request_id);
            // accepted — kick off first poll quickly
            setTimeout(refetchConcepts, 800);
          }
        })
        .catch((err) => {
          console.error("[concepts] network error:", err);
          setConceptsError("Network error — could not reach the server.");
          setConceptsTriggered(false);
        });
    },
    [
      API_BASE,
      ingestedRepoUrl,
      githubToken,
      isGeneratingConcepts,
      refetchConcepts,
    ],

  );

  const docs = useMemo(() => parseDocs(rawDocs), [rawDocs]);
  const features = featuresData?.features ?? [];

  const [isDocsExpanded, setIsDocsExpanded] = useState(true);
  const [isConceptsExpanded, setIsConceptsExpanded] = useState(true);
  const [isImportanceExpanded, setIsImportanceExpanded] = useState(true);
  const [expandedRepoGroups, setExpandedRepoGroups] = useState<
    Record<string, boolean>
  >({});

  // Seed new repo groups (default to expanded), preserve existing toggle state
  useEffect(() => {
    const newKeys = features.reduce<Record<string, boolean>>((acc, f) => {
      const repo = f.repo || getRepoFromFeatureId(f.id);
      if (!(repo in expandedRepoGroups) && !(repo in acc)) {
        acc[repo] = true;
      }
      return acc;
    }, {});
    if (Object.keys(newKeys).length > 0) {
      setExpandedRepoGroups((prev) => ({ ...newKeys, ...prev }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [features]);

  const toggleRepoGroup = (repo: string) => {
    setExpandedRepoGroups((prev) => ({ ...prev, [repo]: !prev[repo] }));
  };

  const groupedConcepts = useMemo(() => {
    const order: string[] = [];
    const map: Record<string, FeatureSummary[]> = {};
    for (const f of features) {
      const repo = f.repo || getRepoFromFeatureId(f.id);
      if (!map[repo]) {
        order.push(repo);
        map[repo] = [];
      }
      map[repo].push(f);
    }
    return order.map((repo) => ({ repo, concepts: map[repo] }));
  }, [features]);

  return (
    <div className="fixed right-0 top-12 bottom-0 w-80 border-l bg-background flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Docs Section */}
        <div>
          <div className="flex items-center">
            <Button
              variant="ghost"
              className="flex-1 justify-between p-2 h-auto"
              onClick={() => setIsDocsExpanded(!isDocsExpanded)}
            >
              <div className="flex items-center gap-2">
                <BookOpen className="h-4 w-4" />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="font-medium cursor-default">Docs</span>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="max-w-56">
                    AI-generated architecture summaries for each repository,
                    built from rules files like .cursorrules and
                    copilot-instructions.md.
                  </TooltipContent>
                </Tooltip>
                <Badge variant="secondary" className="ml-1">
                  {docs.length}
                </Badge>
              </div>
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  isDocsExpanded && "rotate-180",
                )}
              />
            </Button>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="ml-1 shrink-0"
                  onClick={handleGenerateDocs}
                  disabled={isGeneratingDocs}
                  aria-label="Generate docs"
                >
                  {isGeneratingDocs ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Zap className="h-3.5 w-3.5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                Generate / refresh docs
              </TooltipContent>
            </Tooltip>
          </div>

          <AnimatePresence>
            {isDocsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1">
                  {docsStatus && !isGeneratingDocs && (
                    <div
                      className={cn(
                        "mx-1 mb-1 px-3 py-2 rounded-md text-xs",
                        docsStatus.type === "error"
                          ? "bg-destructive/10 text-destructive"
                          : docsStatus.type === "success"
                            ? "bg-primary/10 text-primary"
                            : "bg-muted/40 text-muted-foreground",
                      )}
                    >
                      {docsStatus.message}
                    </div>
                  )}
                  {isDocsLoading ? (
                    <div className="space-y-2 p-2">
                      {[1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className="h-8 bg-muted/30 rounded animate-pulse"
                        />
                      ))}
                    </div>
                  ) : isGeneratingDocs ? (
                    <div className="p-4 text-sm text-muted-foreground text-center flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating docs...
                    </div>
                  ) : docs.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center">
                      No documentation available yet. Use the zap button to generate it.
                    </div>
                  ) : (
                    docs.map((doc) => {
                      const itemKey = `doc-${doc.repoName}`;
                      const isActive = activeItemKey === itemKey;
                      return (
                        <button
                          key={doc.repoName}
                          onClick={() =>
                            onDocClick(doc.repoName, doc.documentation)
                          }
                          className={cn(
                            "w-full text-left p-2 rounded-md text-sm transition-colors",
                            isActive
                              ? "bg-muted/60 font-medium"
                              : "bg-muted/30 hover:bg-muted/50",
                          )}
                        >
                          {doc.repoName}
                        </button>
                      );
                    })
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Concepts Section */}
        <div>
          <div className="flex items-center">
            <Button
              variant="ghost"
              className="flex-1 justify-between p-2 h-auto"
              onClick={() => setIsConceptsExpanded(!isConceptsExpanded)}
            >
              <div className="flex items-center gap-2">
                <Lightbulb className="h-4 w-4" />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="font-medium cursor-default">Concepts</span>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="max-w-56">
                    High-level features derived from your repo's GitHub PR and
                    commit history. Generate to process new PRs and build your
                    concept map.
                  </TooltipContent>
                </Tooltip>
                <Badge variant="secondary" className="ml-1">
                  {features.length}
                </Badge>
              </div>
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  isConceptsExpanded && "rotate-180",
                )}
              />
            </Button>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="ml-1 shrink-0"
                  onClick={handleGenerateConcepts}
                  disabled={isGeneratingConcepts}
                  aria-label="Generate concepts"
                >
                  {isGeneratingConcepts ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Zap className="h-3.5 w-3.5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                Generate / refresh concepts
              </TooltipContent>
            </Tooltip>
          </div>

          <AnimatePresence>
            {isConceptsExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1">
                  {conceptsError && (
                    <div className="mx-1 mb-1 px-3 py-2 rounded-md bg-destructive/10 text-destructive text-xs">
                      {conceptsError}
                    </div>
                  )}
                  {isConceptsLoading ? (
                    <div className="space-y-2 p-2">
                      {[1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className="h-8 bg-muted/30 rounded animate-pulse"
                        />
                      ))}
                    </div>
                  ) : isGeneratingConcepts && features.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating concepts...
                    </div>
                  ) : features.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground text-center">
                      No concepts discovered yet
                    </div>
                  ) : (
                    groupedConcepts.map(({ repo, concepts: group }) => {
                      const shortName = repo.split("/")[1] ?? repo;
                      const isGroupExpanded = expandedRepoGroups[repo] ?? true;
                      return (
                        <div key={repo}>
                          <Button
                            variant="ghost"
                            className="w-full justify-between pl-4 pr-2 py-1 h-auto text-xs text-muted-foreground"
                            onClick={() => toggleRepoGroup(repo)}
                          >
                            <div className="flex items-center gap-1.5">
                              <span className="font-medium">{shortName}</span>
                              <Badge variant="secondary">{group.length}</Badge>
                            </div>
                            <ChevronDown
                              className={cn(
                                "h-3 w-3 transition-transform",
                                isGroupExpanded && "rotate-180",
                              )}
                            />
                          </Button>
                          <AnimatePresence>
                            {isGroupExpanded && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                              >
                                <div className="mt-1 pl-2">
                                  {group.map((concept) => {
                                    const itemKey = `concept-${concept.id}`;
                                    const isActive = activeItemKey === itemKey;
                                    return (
                                      <button
                                        key={concept.id}
                                        onClick={() =>
                                          onConceptClick(
                                            concept.id,
                                            concept.name,
                                            concept.description,
                                          )
                                        }
                                        onMouseEnter={() =>
                                          handleConceptHover(
                                            concept.ref_id || concept.id,
                                          )
                                        }
                                        onMouseLeave={() =>
                                          handleConceptHover(null)
                                        }
                                        className={cn(
                                          "w-full text-left p-2 rounded-md text-sm transition-colors",
                                          isActive
                                            ? "bg-muted/60 font-medium"
                                            : "bg-muted/30 hover:bg-muted/50",
                                        )}
                                      >
                                        {concept.name}
                                      </button>
                                    );
                                  })}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Importance Section */}
        <div>
          <div className="flex items-center">
            <Button
              variant="ghost"
              className="flex-1 justify-between p-2 h-auto"
              onClick={() => setIsImportanceExpanded(!isImportanceExpanded)}
            >
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="font-medium cursor-default">Importance</span>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="max-w-56">
                    Nodes ranked by structural importance — entry points, hubs, and
                    utilities derived from PageRank and graph degree analysis.
                  </TooltipContent>
                </Tooltip>
              </div>
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  isImportanceExpanded && "rotate-180",
                )}
              />
            </Button>
          </div>

          <AnimatePresence>
            {isImportanceExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="mt-2">
                  <ImportanceLens />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
