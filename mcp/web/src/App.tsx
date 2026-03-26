import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { DocViewer, ActiveItem } from "@/components/DocViewer";
import { GraphScene } from "@/graph/GraphScene";

const API_BASE = import.meta.env.VITE_API_BASE || "";

type View = "graph" | "doc";

function App() {
  const [view, setView] = useState<View>("graph");
  const [activeItem, setActiveItem] = useState<ActiveItem | null>(null);

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
    _description: string
  ) => {
    setActiveItem({ type: "concept", id, name, content: "Loading..." });
    setView("doc");

    try {
      const res = await fetch(
        `${API_BASE}/gitree/features/${encodeURIComponent(id)}`
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

  return (
    <div className="flex flex-col h-screen w-full">
      <header className="h-12 border-b border-border flex items-center px-4 shrink-0">
        <button
          onClick={() => setView("graph")}
          className={`text-sm font-semibold tracking-wide hover:text-foreground transition-colors ${
            view === "graph" ? "text-foreground" : "text-muted-foreground"
          }`}
        >
          stakgraph
        </button>
        <span className="text-sm text-muted-foreground ml-2">
          software knowledge graph
        </span>
      </header>
      <div className="flex flex-1 min-h-0">
        <div className="flex-1 pr-80 relative">
          {view === "graph" ? (
            <GraphScene />
          ) : (
            <DocViewer activeItem={activeItem} />
          )}
        </div>
        <Sidebar
          activeItemKey={getActiveItemKey()}
          onDocClick={handleDocClick}
          onConceptClick={handleConceptClick}
        />
      </div>
    </div>
  );
}

export default App;
