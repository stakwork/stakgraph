import { useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { XCircle, X } from "lucide-react";
import { useIngestion } from "@/stores/useIngestion";
import { useSSE } from "@/hooks/useSSE";

const STANDALONE_BASE =
  import.meta.env.VITE_STANDALONE_URL || "http://localhost:7799";

interface IngestionStatusProps {
  onReset: () => void;
}

export function IngestionStatus({ onReset }: IngestionStatusProps) {
  const { phase, currentUpdate, errorMessage, applyUpdate } = useIngestion();

  useSSE(phase === "running" ? `${STANDALONE_BASE}/events` : null, applyUpdate);

  const step = currentUpdate?.step ?? 0;
  const totalSteps = currentUpdate?.total_steps ?? 16;
  const rawPct =
    totalSteps > 0
      ? Math.round(
          ((step - 1 + (currentUpdate?.progress ?? 0) / 100) / totalSteps) *
            100,
        )
      : 0;
  const maxPctRef = useRef(0);
  if (rawPct > maxPctRef.current) maxPctRef.current = rawPct;
  if (phase !== "running") maxPctRef.current = 0;
  const overallPct = maxPctRef.current;
  const description =
    currentUpdate?.step_description ||
    currentUpdate?.message ||
    "Initializing…";

  return (
    <div className="absolute top-3 left-3 z-50 w-72 rounded-xl border border-border bg-background/90 backdrop-blur-sm shadow-lg overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
        {phase === "error" ? (
          <XCircle className="size-3.5 text-destructive shrink-0" />
        ) : (
          <motion.span
            className="size-2 rounded-full bg-primary shrink-0"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ repeat: Infinity, duration: 1.4, ease: "easeInOut" }}
          />
        )}
        <span className="text-xs font-medium flex-1 truncate">
          {phase === "error" ? "Ingestion failed" : "Building graph…"}
        </span>
        {phase === "error" && (
          <button
            onClick={onReset}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      {/* Progress */}
      {phase !== "error" && (
        <div className="px-3 pt-2.5 pb-3 flex flex-col gap-2">
          {/* Bar */}
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-primary"
                animate={{ width: `${overallPct}%` }}
                transition={{ ease: "easeOut", duration: 0.5 }}
              />
            </div>
            <span className="text-[11px] text-muted-foreground tabular-nums w-8 text-right">
              {overallPct}%
            </span>
          </div>

          {/* Step label */}
          <p className="text-[11px] text-muted-foreground truncate transition-all duration-150">
            {step > 0 ? `${step}/${totalSteps} · ` : ""}
            {description}
          </p>

          {/* Stats chips */}
          {currentUpdate?.stats &&
            Object.keys(currentUpdate.stats).length > 0 && (
              <div className="flex flex-wrap gap-1 mt-0.5">
                <AnimatePresence initial={false}>
                  {Object.entries(currentUpdate.stats).map(([k, v]) => (
                    <motion.span
                      key={k}
                      initial={{ opacity: 0, scale: 0.85 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.2 }}
                      className="inline-flex items-center gap-1 text-[10px] rounded-md bg-muted px-1.5 py-0.5 text-muted-foreground"
                    >
                      <span className="capitalize">{k.replace(/_/g, " ")}</span>
                      <motion.span
                        key={v}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.15 }}
                        className="font-mono font-semibold text-foreground"
                      >
                        {v.toLocaleString()}
                      </motion.span>
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
            )}
        </div>
      )}

      {/* Error detail */}
      {phase === "error" && errorMessage && (
        <p className="px-3 py-2 text-[11px] text-destructive">{errorMessage}</p>
      )}
    </div>
  );
}
