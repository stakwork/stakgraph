import { MarkdownRenderer } from "@/components/MarkdownRenderer";

export interface ActiveItem {
  type: "doc" | "concept";
  repoName?: string;
  id?: string;
  name: string;
  content: string;
}

interface DocViewerProps {
  activeItem: ActiveItem | null;
}

export function DocViewer({ activeItem }: DocViewerProps) {
  if (!activeItem) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        Select a document or concept to view
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="prose prose-invert max-w-none">
        <MarkdownRenderer>{activeItem.content}</MarkdownRenderer>
      </div>
    </div>
  );
}
