import { type KeyboardEvent, useCallback, useEffect, useState } from "react";
import { ChevronDown, Layers, Loader2, Search } from "lucide-react";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { useLayerVisibility } from "@/stores/useLayerVisibility";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useApi } from "@/hooks/useApi";
import { cn } from "@/lib/utils";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const SEARCH_LIMIT = 8;

type SearchResult = {
  node_type: string;
  ref_id: string;
  properties: {
    name?: string;
    file?: string;
    [key: string]: unknown;
  };
};

type SearchSortBy = "relevance" | "pagerank";
type EmbeddingsStatusResponse = {
  status: "none" | "partial" | "ready";
  embeddings_count: number;
  eligible_count: number;
  coverage_ratio: number;
};

const SEARCH_QUALITY_LABELS: Record<
  EmbeddingsStatusResponse["status"],
  string
> = {
  none: "None",
  partial: "Partial",
  ready: "Ready",
};

const SEARCH_QUALITY_STYLES: Record<
  EmbeddingsStatusResponse["status"],
  string
> = {
  none: "text-rose-400 bg-rose-500/15 border-rose-500/30",
  partial: "text-amber-300 bg-amber-500/15 border-amber-500/30",
  ready: "text-emerald-300 bg-emerald-500/15 border-emerald-500/30",
};

function formatNodeType(nodeType: string): string {
  return nodeType.replace(/_/g, " ");
}

function shortenPath(filePath?: string): string {
  if (!filePath) return "Path unavailable";
  const normalized = filePath.replace(/\\/g, "/").replace(/^\.?\//, "");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length <= 3) return normalized;
  return `.../${parts.slice(-3).join("/")}`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderHighlightedText(text: string, query: string) {
  const q = query.trim();
  if (!q) return text;

  const regex = new RegExp(`(${escapeRegExp(q)})`, "ig");
  const parts = text.split(regex);

  return parts.map((part, index) =>
    part.toLowerCase() === q.toLowerCase() ? (
      <mark
        key={`${part}-${index}`}
        className="bg-amber-200/70 dark:bg-amber-500/35 text-foreground rounded-sm px-0.5"
      >
        {part}
      </mark>
    ) : (
      <span key={`${part}-${index}`}>{part}</span>
    ),
  );
}

export function LayerTogglePanel() {
  const [searchOpen, setSearchOpen] = useState(true);
  const [layersOpen, setLayersOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [activeResultIndex, setActiveResultIndex] = useState(-1);
  const [selectedNodeTypeFilter, setSelectedNodeTypeFilter] =
    useState<string>("all");
  const [searchSortBy, setSearchSortBy] = useState<SearchSortBy>("relevance");

  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const nodesNormalized = useGraphData((s) => s.nodesNormalized);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);
  const { disabledLayers, toggleLayer } = useLayerVisibility();
  const { data: embeddingsStatus } =
    useApi<EmbeddingsStatusResponse>("/embeddings_status");
  const qualityStatus = embeddingsStatus?.status || "none";

  const embeddingsHintText =
    embeddingsStatus?.status === "partial"
      ? `Embeddings are only partially available (${embeddingsStatus.embeddings_count}/${embeddingsStatus.eligible_count}). Click Enrich to improve semantic search results.`
      : embeddingsStatus?.status === "ready"
        ? `Embeddings are ready (${embeddingsStatus.embeddings_count}/${embeddingsStatus.eligible_count}).`
        : "Embeddings are not available yet. Click Enrich to generate embeddings and improve semantic search results.";

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults([]);
      setSearchError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true);
      setSearchError(null);
      try {
        const params = new URLSearchParams({
          query: q,
          limit: String(SEARCH_LIMIT),
          method: "hybrid",
          output: "json",
          sort_by: searchSortBy,
        });

        const res = await fetch(`${API_BASE}/search?${params.toString()}`, {
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }

        const json = await res.json();
        setResults(Array.isArray(json) ? (json as SearchResult[]) : []);
      } catch (error) {
        if (controller.signal.aborted) return;
        setSearchError(
          error instanceof Error ? error.message : "Search request failed",
        );
        setResults([]);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }, 220);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query, searchSortBy]);

  const resultNodeTypes = Array.from(new Set(results.map((r) => r.node_type)));

  const filteredResults =
    selectedNodeTypeFilter === "all"
      ? results
      : results.filter((result) => result.node_type === selectedNodeTypeFilter);

  useEffect(() => {
    if (
      selectedNodeTypeFilter !== "all" &&
      !resultNodeTypes.includes(selectedNodeTypeFilter)
    ) {
      setSelectedNodeTypeFilter("all");
    }
  }, [resultNodeTypes, selectedNodeTypeFilter]);

  useEffect(() => {
    if (filteredResults.length === 0) {
      setActiveResultIndex(-1);
      return;
    }
    setActiveResultIndex((prev) => {
      if (prev < 0) return 0;
      return Math.min(prev, filteredResults.length - 1);
    });
  }, [filteredResults]);

  const handleSelectResult = useCallback(
    async (refId: string) => {
      let node = nodesNormalized.get(refId);
      if (!node) {
        try {
          const res = await fetch(
            `${API_BASE}/subgraph?ref_id=${encodeURIComponent(refId)}`,
          );
          if (res.ok) {
            const result = await res.json();
            let raw =
              result?.nodes?.find((n: any) => n.ref_id === refId) ||
              result?.node;
            if (!raw && Array.isArray(result)) {
              raw = result[0];
            } else if (!raw) {
              raw = result;
            }
            if (raw) {
              const extended = {
                ...raw,
                x: 0,
                y: 0,
                z: 0,
                sources: [],
                targets: [],
                index: -1,
              };
              useGraphData.getState().nodesNormalized.set(refId, extended);
              node = extended;
            }
          }
        } catch (error) {
          console.error("[search-select]", error);
        }
      }

      if (node) {
        setSelectedNode(node);
      }
    },
    [nodesNormalized, setSelectedNode],
  );

  const handleSearchInputKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (filteredResults.length === 0) return;

      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveResultIndex((prev) =>
          prev >= filteredResults.length - 1 ? 0 : prev + 1,
        );
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveResultIndex((prev) =>
          prev <= 0 ? filteredResults.length - 1 : prev - 1,
        );
      } else if (event.key === "Enter") {
        event.preventDefault();
        const idx = activeResultIndex >= 0 ? activeResultIndex : 0;
        const target = filteredResults[idx];
        if (target) {
          void handleSelectResult(target.ref_id);
        }
      }
    },
    [activeResultIndex, filteredResults, handleSelectResult],
  );

  return (
    <div className="absolute left-3 top-4 z-20 w-72 pointer-events-auto">
      <div className="rounded-xl border border-border bg-background/92 backdrop-blur-md shadow-lg overflow-hidden">
        <Collapsible open={searchOpen} onOpenChange={setSearchOpen}>
          <CollapsibleTrigger className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-muted/40 transition-colors">
            <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <Search className="size-3.5" />
              Search
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={cn(
                      "inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide",
                      SEARCH_QUALITY_STYLES[qualityStatus],
                    )}
                  >
                    {SEARCH_QUALITY_LABELS[qualityStatus]}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-64">
                  {embeddingsHintText}
                </TooltipContent>
              </Tooltip>
            </span>
            <ChevronDown
              className={cn(
                "size-4 text-muted-foreground transition-transform",
                searchOpen && "rotate-180",
              )}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-3 pb-3 space-y-2.5">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleSearchInputKeyDown}
                placeholder="Search code graph..."
                className="w-full h-8 rounded-md border border-border bg-background px-2.5 text-xs text-foreground outline-none focus:ring-2 focus:ring-ring/40"
              />

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setSearchSortBy("relevance")}
                  className={cn(
                    "text-[11px] rounded px-2 py-1 border transition-colors",
                    searchSortBy === "relevance"
                      ? "bg-foreground text-background border-foreground"
                      : "bg-muted/40 text-muted-foreground border-border hover:bg-muted/60",
                  )}
                >
                  Relevance
                </button>
                <button
                  onClick={() => setSearchSortBy("pagerank")}
                  className={cn(
                    "text-[11px] rounded px-2 py-1 border transition-colors",
                    searchSortBy === "pagerank"
                      ? "bg-foreground text-background border-foreground"
                      : "bg-muted/40 text-muted-foreground border-border hover:bg-muted/60",
                  )}
                >
                  PageRank
                </button>
              </div>
              {resultNodeTypes.length > 0 && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
                  <button
                    onClick={() => setSelectedNodeTypeFilter("all")}
                    className={cn(
                      "shrink-0 text-[11px] rounded-full px-2 py-1 border transition-colors",
                      selectedNodeTypeFilter === "all"
                        ? "bg-foreground text-background border-foreground"
                        : "bg-muted/40 text-muted-foreground border-border hover:bg-muted/60",
                    )}
                  >
                    All
                  </button>
                  {resultNodeTypes.map((nodeType) => (
                    <button
                      key={nodeType}
                      onClick={() => setSelectedNodeTypeFilter(nodeType)}
                      className={cn(
                        "shrink-0 text-[11px] rounded-full px-2 py-1 border transition-colors",
                        selectedNodeTypeFilter === nodeType
                          ? "bg-foreground text-background border-foreground"
                          : "bg-muted/40 text-muted-foreground border-border hover:bg-muted/60",
                      )}
                    >
                      {formatNodeType(nodeType)}
                    </button>
                  ))}
                </div>
              )}

              <div className="max-h-72 overflow-y-auto space-y-1">
                {loading && (
                  <div className="text-xs text-muted-foreground px-2 py-1.5 flex items-center gap-2">
                    <Loader2 className="size-3.5 animate-spin" />
                    Searching...
                  </div>
                )}

                {!loading && searchError && (
                  <div className="text-xs text-destructive px-2 py-1.5 rounded-md bg-destructive/10">
                    {searchError}
                  </div>
                )}

                {!loading &&
                  !searchError &&
                  query.trim().length > 0 &&
                  filteredResults.length === 0 && (
                    <div className="text-xs text-muted-foreground px-2 py-1.5">
                      No matches found
                    </div>
                  )}

                {!loading && !searchError && query.trim().length === 0 && (
                  <div className="text-xs text-muted-foreground px-2 py-1.5">
                    Type to search nodes by meaning and keywords
                  </div>
                )}

                {!loading &&
                  !searchError &&
                  filteredResults.map((result, index) => {
                    const name = result.properties.name || result.ref_id;
                    const path = shortenPath(result.properties.file);
                    const isActive = index === activeResultIndex;

                    return (
                      <button
                        key={result.ref_id}
                        onMouseEnter={() => setActiveResultIndex(index)}
                        onClick={() => void handleSelectResult(result.ref_id)}
                        className={cn(
                          "w-full text-left rounded-md border px-2.5 py-2 transition-colors",
                          isActive
                            ? "border-border bg-muted/70"
                            : "border-transparent bg-muted/35 hover:bg-muted/55 hover:border-border",
                        )}
                      >
                        <div className="text-xs font-medium text-foreground truncate">
                          {renderHighlightedText(name, query)}
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-[11px]">
                          <span className="inline-flex rounded-sm bg-muted px-1.5 py-0.5 text-muted-foreground">
                            {formatNodeType(result.node_type)}
                          </span>
                          <span className="text-muted-foreground truncate">
                            {renderHighlightedText(path, query)}
                          </span>
                        </div>
                      </button>
                    );
                  })}
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="h-px bg-border" />

        <Collapsible open={layersOpen} onOpenChange={setLayersOpen}>
          <CollapsibleTrigger className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-muted/40 transition-colors">
            <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <Layers className="size-3.5" />
              Layers
            </span>
            <ChevronDown
              className={cn(
                "size-4 text-muted-foreground transition-transform",
                layersOpen && "rotate-180",
              )}
            />
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-3 pb-3 flex flex-col gap-1">
              {nodeTypes.length === 0 && (
                <div className="text-xs text-muted-foreground px-1 py-1.5">
                  No layers available
                </div>
              )}
              {nodeTypes.map((nodeType) => {
                const color = getColorForType(nodeType);
                const disabled = disabledLayers.has(nodeType);
                return (
                  <button
                    key={nodeType}
                    onClick={() => toggleLayer(nodeType)}
                    className="flex items-center gap-2.5 px-1 py-1 rounded-md hover:bg-muted/50 transition-colors text-left w-full"
                  >
                    <span
                      className="w-3 h-3 rounded-sm shrink-0 border transition-all"
                      style={{
                        backgroundColor: disabled ? "transparent" : color,
                        borderColor: color,
                        opacity: disabled ? 0.5 : 1,
                      }}
                    />
                    <span
                      className="text-xs transition-colors"
                      style={{
                        color: disabled ? "var(--muted-foreground)" : color,
                        textDecoration: disabled ? "line-through" : "none",
                        opacity: disabled ? 0.5 : 1,
                      }}
                    >
                      {formatNodeType(nodeType)}
                    </span>
                  </button>
                );
              })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  );
}
