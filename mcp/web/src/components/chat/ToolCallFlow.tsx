import { motion, AnimatePresence } from "framer-motion";
import type { ToolCallEvent } from "@/stores/useChat";
import {
  Search,
  FileText,
  Terminal,
  Globe,
  FolderTree,
  Cpu,
  type LucideIcon,
} from "lucide-react";

const TOOL_ICONS: Record<string, LucideIcon> = {
  repo_overview: FolderTree,
  file_summary: FileText,
  bash: Terminal,
  fulltext_search: Search,
  vector_search: Search,
  web_search: Globe,
  recent_commits: Terminal,
  recent_contributions: Terminal,
  list_concepts: FolderTree,
  learn_concept: FileText,
};

/** Tools that should always show a label (they have no useful args to preview) */
const TOOL_LABELS: Record<string, string> = {
  repo_overview: "Scanning repo structure",
  list_concepts: "Listing concepts",
  learn_concept: "Learning concept",
};

function getToolIcon(toolName: string): LucideIcon {
  return TOOL_ICONS[toolName] || Cpu;
}

/** Render a short preview of tool input */
function inputPreview(input: unknown): string | null {
  if (!input || typeof input !== "object") return null;
  const obj = input as Record<string, unknown>;
  // common patterns
  if (typeof obj.concept_id === "string") return obj.concept_id;
  if (typeof obj.path === "string") return obj.path;
  if (typeof obj.query === "string") return obj.query;
  if (typeof obj.command === "string") return obj.command;
  if (typeof obj.file_path === "string") return obj.file_path;
  return null;
}

interface ToolCallFlowProps {
  toolCalls: ToolCallEvent[];
  isActive: boolean;
}

export function ToolCallFlow({ toolCalls, isActive }: ToolCallFlowProps) {
  if (toolCalls.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="pointer-events-auto flex justify-center w-full"
    >
      <div className="max-w-[70vw] sm:max-w-[450px] md:max-w-[500px] lg:max-w-[600px] w-full">
        <div className="rounded-2xl px-4 py-3 shadow-sm backdrop-blur-sm bg-muted/10 space-y-1.5">
          <AnimatePresence mode="popLayout">
            {toolCalls.map((tc, i) => {
              const Icon = getToolIcon(tc.toolName);
              const preview = inputPreview(tc.input);
              const isLast = i === toolCalls.length - 1;

              return (
                <motion.div
                  key={tc.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: isLast ? 1 : 0.5, x: 0 }}
                  transition={{ duration: 0.2 }}
                  className="flex items-center gap-2 text-sm"
                >
                  <Icon className="size-3.5 text-muted-foreground shrink-0" />
                  {TOOL_LABELS[tc.toolName] && (
                    <span className="text-foreground/70">{TOOL_LABELS[tc.toolName]}</span>
                  )}
                  {preview && (
                    <span className="text-muted-foreground truncate text-xs font-mono">
                      {preview}
                    </span>
                  )}
                  {isLast && isActive && <PulsingDots />}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}

function PulsingDots() {
  return (
    <span className="flex gap-0.5 ml-1">
      {[0, 0.2, 0.4].map((delay) => (
        <motion.span
          key={delay}
          className="text-muted-foreground"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1.2, repeat: Infinity, delay }}
        >
          .
        </motion.span>
      ))}
    </span>
  );
}
