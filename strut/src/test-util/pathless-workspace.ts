import { FileWorkspaceStore, type WorkspaceStore } from "../workspace.js";

/** A `WorkspaceStore` that is NOT a `FileWorkspaceStore` and has no `path`:
 *  the file impl's methods bound onto a plain object. It stands in for a
 *  non-file backend, proving nothing reaches through to a directory. */
export function pathlessWorkspace(inner: FileWorkspaceStore): WorkspaceStore {
  const out: Record<string, unknown> = {};
  for (const k of Object.getOwnPropertyNames(FileWorkspaceStore.prototype)) {
    if (k === "constructor") continue;
    const v = (inner as unknown as Record<string, unknown>)[k];
    if (typeof v === "function") out[k] = v.bind(inner);
  }
  return out as unknown as WorkspaceStore;
}

