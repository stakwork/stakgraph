import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers } from "lucide-react";
import { useGraphData, getColorForType } from "@/stores/useGraphData";
import { useLayerVisibility } from "@/stores/useLayerVisibility";

export function LayerTogglePanel() {
  const [open, setOpen] = useState(false);
  const nodeTypes = useGraphData((s) => s.nodeTypes);
  const { disabledLayers, toggleLayer } = useLayerVisibility();

  if (nodeTypes.length === 0) return null;

  return (
    <div
      className="absolute left-0 top-1/2 -translate-y-1/2 z-20 flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {/* Invisible hover trigger strip on left edge */}
      <div className="w-3 h-48 cursor-pointer" />

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -20, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="ml-0 bg-background/90 backdrop-blur-sm border border-border rounded-r-xl shadow-xl py-3 px-3 min-w-40"
          >
            <div className="flex items-center gap-2 mb-2.5 px-1">
              <Layers className="size-3.5 text-muted-foreground" />
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Layers
              </span>
            </div>
            <div className="flex flex-col gap-1">
              {nodeTypes.map((nodeType) => {
                const color = getColorForType(nodeType);
                const disabled = disabledLayers.has(nodeType);
                return (
                  <button
                    key={nodeType}
                    onClick={() => toggleLayer(nodeType)}
                    className="flex items-center gap-2.5 px-1 py-1 rounded-md hover:bg-muted/50 transition-colors text-left w-full group"
                  >
                    {/* Color swatch / checkbox */}
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
                      {nodeType.replace(/_/g, " ")}
                    </span>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
