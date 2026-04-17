import { useState, useEffect, useMemo, useCallback } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { apiFetch } from "@/lib/api";

const API_BASE = import.meta.env.VITE_API_BASE || "";

interface ImportanceNode {
  ref_id: string;
  name: string;
  file: string;
  label: string;
  pagerank: number;
  in_degree: number;
  out_degree: number;
  importance_tag: string | null;
}

interface TagResponse {
  tag: string;
  nodes: ImportanceNode[];
  total: number;
}

const TABS = [
  { key: "entry_point", label: "Entry Points", color: "#FFD700" },
  { key: "hub", label: "Hubs", color: "#FF4444" },
  { key: "utility", label: "Utilities", color: "#00BCD4" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

function groupByNodeType(nodes: ImportanceNode[]): { nodeType: string; nodes: ImportanceNode[] }[] {
  const order: string[] = [];
  const map: Record<string, ImportanceNode[]> = {};
  for (const n of nodes) {
    if (!map[n.label]) {
      order.push(n.label);
      map[n.label] = [];
    }
    map[n.label].push(n);
  }
  return order.map((nodeType) => ({ nodeType, nodes: map[nodeType] }));
}

export function ImportanceLens() {
  const [activeTab, setActiveTab] = useState<TabKey>("entry_point");
  const [data, setData] = useState<Record<string, ImportanceNode[]>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

  const nodesNormalized = useGraphData((s) => s.nodesNormalized);
  const setSelectedNode = useGraphData((s) => s.setSelectedNode);
  const setImportanceFilter = useGraphData((s) => s.setImportanceFilter);
  const importanceFilter = useGraphData((s) => s.importanceFilter);

  const fetchTag = useCallback(async (tag: string) => {
    if (data[tag]) return;
    setLoading((p) => ({ ...p, [tag]: true }));
    try {
      const res = await apiFetch(`${API_BASE}/importance/tag?tag=${tag}&limit=200`);
      if (res.ok) {
        const json: TagResponse = await res.json();
        setData((p) => ({ ...p, [tag]: json.nodes }));
      }
    } catch (e) {
      console.error(`[importance] fetch ${tag} error:`, e);
    } finally {
      setLoading((p) => ({ ...p, [tag]: false }));
    }
  }, [data]);

  useEffect(() => {
    fetchTag(activeTab);
  }, [activeTab, fetchTag]);

  const currentNodes = data[activeTab] || [];
  const grouped = useMemo(() => groupByNodeType(currentNodes), [currentNodes]);
  const isLoading = loading[activeTab];
  const activeTabMeta = TABS.find((t) => t.key === activeTab)!;

  const toggleGroup = (nodeType: string) => {
    setExpandedGroups((p) => ({ ...p, [nodeType]: !(p[nodeType] ?? true) }));
  };

  const handleNodeClick = useCallback(
    async (refId: string) => {
      console.log("refId:", refId);
      console.log("in nodesNormalized:", nodesNormalized.has(refId));
      let node = nodesNormalized.get(refId);
      if (!node) {
        try {
          const res = await apiFetch(
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
        } catch (e) {
          console.error("[handleNodeClick] fetch failed", e);
        }
      }

      if (node) setSelectedNode(node);
    },
    [nodesNormalized, setSelectedNode],
  );

  const handleFilterToggle = useCallback(
    (tag: string, nodeType?: string) => {
      const { importanceFilter: current } = useGraphData.getState();
      if (current.tag === tag && current.nodeType === (nodeType || null)) {
        setImportanceFilter(null);
      } else {
        setImportanceFilter(tag, nodeType || null);
      }
    },
    [setImportanceFilter],
  );

  return (
    <div>
      <div style={{ display: "flex", gap: 2, marginBottom: 8 }}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "flex-1 text-xs py-1.5 px-2 rounded-md transition-colors font-medium",
                isActive
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground/80",
              )}
              style={{
                background: isActive ? `${tab.color}15` : "transparent",
                borderBottom: isActive ? `2px solid ${tab.color}` : "2px solid transparent",
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div style={{ marginBottom: 6 }}>
        <button
          onClick={() => handleFilterToggle(activeTab)}
          className="text-xs px-2 py-1 rounded transition-colors"
          style={{
            background:
              importanceFilter.tag === activeTab && !importanceFilter.nodeType
                ? `${activeTabMeta.color}25`
                : "transparent",
            border: `1px solid ${importanceFilter.tag === activeTab && !importanceFilter.nodeType ? activeTabMeta.color + "60" : "rgba(255,255,255,0.1)"}`,
            color:
              importanceFilter.tag === activeTab && !importanceFilter.nodeType
                ? activeTabMeta.color
                : "var(--muted-foreground)",
          }}
        >
          {importanceFilter.tag === activeTab && !importanceFilter.nodeType
            ? "✓ Highlighting all"
            : "Highlight all on graph"}
        </button>
      </div>

      {isLoading ? (
        <div className="p-4 text-sm text-muted-foreground text-center flex items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading...
        </div>
      ) : currentNodes.length === 0 ? (
        <div className="p-4 text-sm text-muted-foreground text-center">
          No nodes scored yet. Run importance scoring first.
        </div>
      ) : (
        grouped.map(({ nodeType, nodes }) => {
          const isExpanded = expandedGroups[nodeType] ?? true;
          const isTypeFiltered =
            importanceFilter.tag === activeTab &&
            importanceFilter.nodeType === nodeType;
          return (
            <div key={nodeType}>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Button
                  variant="ghost"
                  className="flex-1 justify-between pl-2 pr-2 py-1 h-auto text-xs"
                  onClick={() => toggleGroup(nodeType)}
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: getColorForType(nodeType),
                        flexShrink: 0,
                      }}
                    />
                    <span className="font-medium">{nodeType}</span>
                    <Badge variant="secondary">{nodes.length}</Badge>
                  </div>
                  <ChevronDown
                    className={cn(
                      "h-3 w-3 transition-transform",
                      isExpanded && "rotate-180",
                    )}
                  />
                </Button>
                <button
                  onClick={() => handleFilterToggle(activeTab, nodeType)}
                  className="text-xs px-1.5 py-0.5 rounded shrink-0 transition-colors"
                  style={{
                    background: isTypeFiltered
                      ? `${activeTabMeta.color}25`
                      : "transparent",
                    border: `1px solid ${isTypeFiltered ? activeTabMeta.color + "60" : "rgba(255,255,255,0.08)"}`,
                    color: isTypeFiltered
                      ? activeTabMeta.color
                      : "var(--muted-foreground)",
                    fontSize: 9,
                  }}
                  title={
                    isTypeFiltered
                      ? "Clear filter"
                      : `Highlight ${nodeType} ${activeTab}s`
                  }
                >
                  {isTypeFiltered ? "✓" : "⊙"}
                </button>
              </div>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-1 pl-2 space-y-0.5">
                      {nodes.map((n) => (
                        <button
                          key={n.ref_id}
                          onClick={() => handleNodeClick(n.ref_id)}
                          className="w-full text-left p-1.5 rounded-md text-xs bg-muted/30 hover:bg-muted/50 transition-colors flex items-center gap-2"
                        >
                          <span
                            className="truncate flex-1 shrink-0"
                            title={n.name}
                          >
                            {n.name}
                          </span>
                          <span className="truncate flex-2  text-neutral-700 dark:text-neutral-400">
                            {n.file.split("/").slice(2).join("/")}
                          </span>
                          <span
                            className="text-muted-foreground shrink-0"
                            style={{ fontSize: 9 }}
                          >
                            {n.pagerank.toFixed(4)}
                          </span>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })
      )}
    </div>
  );
}
