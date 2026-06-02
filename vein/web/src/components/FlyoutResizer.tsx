import { useCallback, useEffect, useRef, useState } from "preact/hooks";

const MIN_WIDTH = 320;
const MIN_CANVAS = 360; // keep at least this much room to the left
const DEFAULT_WIDTH = 420;
const STORAGE_KEY = "vein.flyoutWidth";

function getShell(): HTMLElement | null {
  return document.querySelector(".shell");
}

function applyWidth(px: number) {
  const shell = getShell();
  if (!shell) return;
  shell.style.setProperty("--flyout-width", `${px}px`);
}

// Restore persisted width once on module load
let restored = false;
function restoreOnce() {
  if (restored) return;
  restored = true;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      if (!isNaN(n)) applyWidth(n);
    }
  } catch { /* ignore */ }
}

export function FlyoutResizer() {
  const [dragging, setDragging] = useState(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  useEffect(() => { restoreOnce(); }, []);

  const onPointerDown = useCallback((e: PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const flyout = (e.currentTarget as HTMLElement).closest(".flyout");
    const currentPx = flyout ? flyout.getBoundingClientRect().width : DEFAULT_WIDTH;
    startX.current = e.clientX;
    startWidth.current = currentPx;
    setDragging(true);
    document.body.classList.add("is-resizing-flyout");

    const onMove = (ev: PointerEvent) => {
      const delta = startX.current - ev.clientX; // drag left = wider
      const maxW = Math.max(MIN_WIDTH, window.innerWidth - MIN_CANVAS);
      const next = Math.min(maxW, Math.max(MIN_WIDTH, startWidth.current + delta));
      applyWidth(next);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("is-resizing-flyout");
      setDragging(false);
      const shell = getShell();
      if (shell) {
        const w = getComputedStyle(shell).getPropertyValue("--flyout-width").trim();
        const n = parseInt(w, 10);
        if (!isNaN(n)) {
          try { localStorage.setItem(STORAGE_KEY, n.toString()); } catch { /* ignore */ }
        }
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  const onDoubleClick = useCallback(() => {
    applyWidth(DEFAULT_WIDTH);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  return (
    <div
      class={`flyout-resizer ${dragging ? "is-dragging" : ""}`}
      onPointerDown={onPointerDown}
      onDblClick={onDoubleClick}
      title="Drag to resize · double-click to reset"
      role="separator"
      aria-orientation="vertical"
    />
  );
}
