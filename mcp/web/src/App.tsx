import { useState, useEffect } from "react";
import { Plus } from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { DocViewer, ActiveItem } from "@/components/DocViewer";
import { GraphScene } from "@/graph/GraphScene";
import { Onboarding } from "@/components/Onboarding";
import { IngestionStatus } from "@/components/IngestionStatus";
import { LayerTogglePanel } from "@/components/LayerTogglePanel";
import { Chat } from "@/components/chat/Chat";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";
import { SettingsToggle } from "@/components/chat/Settings";
import { SyncButton } from "@/components/SyncButton";
import { Toaster } from "@/components/ui/sonner";

const API_BASE = import.meta.env.VITE_API_BASE || "";

type View = "graph" | "doc" | "onboarding" | null;

function App() {
  const [view, setView] = useState<View>(null);
  const [activeItem, setActiveItem] = useState<ActiveItem | null>(null);
  const { phase, reset: resetIngestion } = useIngestion();
  const { reset: resetGraph } = useGraphData();
  // Preflight: check if Neo4j already has graph data before deciding which view to show
  useEffect(() => {
    let cancelled = false;
    async function checkGraph() {
      try {
        const res = await fetch(`${API_BASE}/graph?limit=1&no_body=true`);
        if (cancelled) return;
        if (res.ok) {
          const data = await res.json();
          const hasNodes = Array.isArray(data?.nodes) && data.nodes.length > 0;
          setView(hasNodes ? "graph" : "onboarding");
        } else {
          setView("onboarding");
        }
      } catch {
        if (!cancelled) setView("onboarding");
      }
    }
    checkGraph();
    return () => {
      cancelled = true;
    };
  }, []);

  // When ingestion completes, reload graph data
  useEffect(() => {
    if (phase === "complete") {
      resetGraph();
      setView("graph");
    }
  }, [phase, resetGraph]);

  const handleDocClick = (repoName: string, documentation: string) => {
    setActiveItem({
      type: "doc",
      repoName,
      name: repoName,
      content: documentation,
    });
    setView("doc");
  };

  const handleConceptClick = async (
    id: string,
    name: string,
    _description: string,
  ) => {
    setActiveItem({ type: "concept", id, name, content: "Loading..." });
    setView("doc");

    try {
      const res = await fetch(
        `${API_BASE}/gitree/features/${encodeURIComponent(id)}`,
      );
      if (res.ok) {
        const data = await res.json();
        const documentation =
          data?.feature?.documentation ||
          data?.feature?.description ||
          _description;
        setActiveItem({ type: "concept", id, name, content: documentation });
      }
    } catch (error) {
      console.error("Failed to fetch concept documentation:", error);
      setActiveItem({ type: "concept", id, name, content: _description });
    }
  };

  const getActiveItemKey = () => {
    if (view !== "doc" || !activeItem) return null;
    if (activeItem.type === "doc") return `doc-${activeItem.repoName}`;
    return `concept-${activeItem.id}`;
  };

  const showSidebar = view === "graph" || view === "doc";
  const ingesting = phase === "running" || phase === "error";

  if (view === null) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <div className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-full">
      <header className="h-12 border-b border-border flex items-center px-4 shrink-0 gap-3">
        <button
          onClick={() => setView("graph")}
          className={`flex items-center gap-2 text-sm font-semibold tracking-wide hover:text-foreground transition-colors ${
            view === "graph" ? "text-foreground" : "text-muted-foreground"
          }`}
        >
          <img src="/favicon.ico" alt="" className="w-5 h-5" />
          stakgraph
        </button>
        <span className="text-sm text-muted-foreground">
          software knowledge graph
        </span>
        <div className="ml-auto flex items-center gap-2">
          {!ingesting && view !== "onboarding" && (
            <button
              onClick={() => {
                resetIngestion();
                setView("onboarding");
              }}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Plus className="size-3.5" />
              Add repository
            </button>
          )}
          <SyncButton />
          <SettingsToggle />
        </div>
      </header>
      <div className="flex flex-1 min-h-0 relative">
        <div className={`flex-1 relative ${showSidebar ? "pr-80" : ""}`}>
          {/* Keep GraphScene mounted but hidden so it doesn't re-fetch/re-render */}
          {(view === "graph" || view === "doc") && (
            <div
              className={`absolute inset-0 ${showSidebar ? "right-80" : ""}`}
              style={{ visibility: view === "graph" ? "visible" : "hidden" }}
            >
              <GraphScene />
              <LayerTogglePanel />
            </div>
          )}
          {view === "doc" && (
            <div className={`absolute inset-0 ${showSidebar ? "right-80" : ""} bg-background`}>
              <DocViewer
                activeItem={activeItem}
                onClose={() => setView("graph")}
              />
            </div>
          )}
          {view === "onboarding" && (
            <Onboarding onStarted={() => setView("graph")} />
          )}
        </div>
        {view === "graph" && (
          <div
            className={`pointer-events-none absolute inset-0 ${showSidebar ? "right-80" : ""}`}
          >
            <Chat />
          </div>
        )}
        {showSidebar && (
          <Sidebar
            activeItemKey={getActiveItemKey()}
            onDocClick={handleDocClick}
            onConceptClick={handleConceptClick}
          />
        )}
        {ingesting && (
          <IngestionStatus
            onReset={() => {
              resetIngestion();
              setView("onboarding");
            }}
          />
        )}
      </div>
      <Toaster />
    </div>
  );
}

export default App;
