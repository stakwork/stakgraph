import { create } from "zustand";

interface LayerVisibilityState {
  disabledLayers: Set<string>;
  toggleLayer: (nodeType: string) => void;
  isVisible: (nodeType: string) => boolean;
}

export const useLayerVisibility = create<LayerVisibilityState>((set, get) => ({
  disabledLayers: new Set(),

  toggleLayer: (nodeType: string) => {
    const next = new Set(get().disabledLayers);
    if (next.has(nodeType)) {
      next.delete(nodeType);
    } else {
      next.add(nodeType);
    }
    set({ disabledLayers: next });
  },

  isVisible: (nodeType: string) => !get().disabledLayers.has(nodeType),
}));
