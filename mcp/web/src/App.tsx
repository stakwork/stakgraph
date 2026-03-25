import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { DocViewer, ActiveItem } from "@/components/DocViewer";

const API_BASE = import.meta.env.VITE_API_BASE || "";

function App() {
  const [activeItem, setActiveItem] = useState<ActiveItem | null>(null);

  const handleDocClick = (repoName: string, documentation: string) => {
    setActiveItem({
      type: "doc",
      repoName,
      name: repoName,
      content: documentation,
    });
  };

  const handleConceptClick = async (id: string, name: string, _description: string) => {
    // Show immediately with description, then fetch full documentation
    setActiveItem({ type: "concept", id, name, content: "Loading..." });

    try {
      const res = await fetch(
        `${API_BASE}/gitree/features/${encodeURIComponent(id)}`
      );
      if (res.ok) {
        const data = await res.json();
        const documentation = data?.feature?.documentation || data?.feature?.description || _description;
        setActiveItem({ type: "concept", id, name, content: documentation });
      }
    } catch (error) {
      console.error("Failed to fetch concept documentation:", error);
      setActiveItem({ type: "concept", id, name, content: _description });
    }
  };

  const getActiveItemKey = () => {
    if (!activeItem) return null;
    if (activeItem.type === "doc") return `doc-${activeItem.repoName}`;
    return `concept-${activeItem.id}`;
  };

  return (
    <div className="flex flex-col h-screen w-full">
      <header className="h-12 border-b border-border flex items-center px-4 shrink-0">
        <span className="text-sm font-semibold tracking-wide">stakgraph</span>
        <span className="text-sm text-muted-foreground ml-2">software knowledge graph</span>
      </header>
      <div className="flex flex-1 min-h-0">
        <div className="flex-1 pr-80">
          <DocViewer activeItem={activeItem} />
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
