import { useState, useEffect } from "react";
import { X, Eye, Loader2 } from "lucide-react";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { ProvenanceTree } from "@/components/ProvenanceTree";
import type { ProvenanceData } from "@/components/ProvenanceTree";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export interface ActiveItem {
  type: "doc" | "concept";
  repoName?: string;
  id?: string;
  name: string;
  content: string;
}

interface DocViewerProps {
  activeItem: ActiveItem | null;
  onClose?: () => void;
}

export function DocViewer({ activeItem, onClose }: DocViewerProps) {
  const [isProvenanceOpen, setIsProvenanceOpen] = useState(false);
  const [provenanceData, setProvenanceData] = useState<ProvenanceData | null>(null);
  const [provenanceLoading, setProvenanceLoading] = useState(false);

  const isConcept = activeItem?.type === "concept";
  const conceptId = activeItem?.id;

  // Fetch provenance when a concept is viewed
  useEffect(() => {
    if (!isConcept || !conceptId) {
      setProvenanceData(null);
      setIsProvenanceOpen(false);
      return;
    }

    let cancelled = false;
    async function fetchProvenance() {
      setProvenanceLoading(true);
      try {
        const res = await fetch(`${API_BASE}/gitree/provenance`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conceptIds: [conceptId] }),
        });
        if (cancelled) return;
        if (res.ok) {
          const data = await res.json();
          setProvenanceData(data);
        }
      } catch (error) {
        console.error("Failed to fetch provenance:", error);
      } finally {
        if (!cancelled) setProvenanceLoading(false);
      }
    }
    fetchProvenance();
    return () => { cancelled = true; };
  }, [isConcept, conceptId]);

  const hasProvenance = provenanceData?.concepts?.some(
    (c) => c.files && c.files.length > 0
  );

  if (!activeItem) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Select a document or concept to view
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* Provenance sidebar */}
      {isProvenanceOpen && provenanceData && (
        <div className="w-80 shrink-0 border-r border-border bg-background overflow-y-auto p-4">
          <ProvenanceTree provenanceData={provenanceData} />
        </div>
      )}

      {/* Main doc content */}
      <div className="flex-1 overflow-y-auto p-6 min-w-0">
        <div className="flex justify-end mb-2 gap-1">
          {isConcept && (hasProvenance || provenanceLoading) && (
            <button
              onClick={() => setIsProvenanceOpen(!isProvenanceOpen)}
              className={`p-1.5 rounded-md transition-colors ${
                isProvenanceOpen
                  ? "text-primary bg-primary/10 hover:bg-primary/20"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
              aria-label="Toggle provenance sources"
              title="View knowledge sources"
            >
              {provenanceLoading ? (
                <Loader2 className="size-5 animate-spin" />
              ) : (
                <Eye className="size-5" />
              )}
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              aria-label="Close"
            >
              <X className="size-5" />
            </button>
          )}
        </div>
        <div className="prose prose-invert max-w-none">
          <MarkdownRenderer>{activeItem.content}</MarkdownRenderer>
        </div>
      </div>
    </div>
  );
}
