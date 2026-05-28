import { useCallback, useEffect, useRef, useState } from "preact/hooks";

const MIN_HEIGHT = 80;
const MAX_HEIGHT_OFFSET = 200; // keep at least this much above for canvas + topbar
const STORAGE_KEY = "vein.eventsHeight";

function getShell(): HTMLElement | null {
  return document.querySelector(".shell");
}

function applyHeight(px: number) {
  const shell = getShell();
  if (!shell) return;
  shell.style.setProperty("--events-height", `${px}px`);
}

// Restore persisted height once on module load
let restored = false;
function restoreOnce() {
  if (restored) return;
  restored = true;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      if (!isNaN(n)) applyHeight(n);
    }
  } catch { /* ignore */ }
}

export function EventsResizer() {
  const [dragging, setDragging] = useState(false);
  const startY = useRef(0);
  const startHeight = useRef(0);

  useEffect(() => { restoreOnce(); }, []);

  const onPointerDown = useCallback((e: PointerEvent) => {
    e.preventDefault();
    const shell = getShell();
    if (!shell) return;
    const current = getComputedStyle(shell).getPropertyValue("--events-height").trim();
    const currentPx = parseInt(current, 10) || 220;
    startY.current = e.clientY;
    startHeight.current = currentPx;
    setDragging(true);
    document.body.classList.add("is-resizing-events");

    const onMove = (ev: PointerEvent) => {
      const delta = startY.current - ev.clientY; // up = larger
      const maxH = Math.max(MIN_HEIGHT, window.innerHeight - MAX_HEIGHT_OFFSET);
      const next = Math.min(maxH, Math.max(MIN_HEIGHT, startHeight.current + delta));
      applyHeight(next);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("is-resizing-events");
      setDragging(false);
      const shell = getShell();
      if (shell) {
        const h = getComputedStyle(shell).getPropertyValue("--events-height").trim();
        try { localStorage.setItem(STORAGE_KEY, parseInt(h, 10).toString()); } catch { /* ignore */ }
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  const onDoubleClick = useCallback(() => {
    applyHeight(220);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  return (
    <div
      class={`events-resizer ${dragging ? "is-dragging" : ""}`}
      onPointerDown={onPointerDown}
      onDblClick={onDoubleClick}
      title="Drag to resize · double-click to reset"
      role="separator"
      aria-orientation="horizontal"
    />
  );
}
