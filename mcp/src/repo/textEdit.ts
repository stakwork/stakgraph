import {
  existsSync,
  statSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
} from "node:fs";
import { resolve, isAbsolute, join, sep, dirname } from "node:path";

/** Max chars returned by a `view` before truncation. */
export const FILE_VIEW_MAX_CHARS = 200_000;

/** The Anthropic text-editor tool's input shape (also used by the generic
 *  fallback for non-anthropic providers). All commands operate on a path that
 *  MUST resolve inside `cwd`. */
export interface TextEditInput {
  command: "view" | "create" | "str_replace" | "insert";
  path: string;
  file_text?: string;
  insert_line?: number;
  new_str?: string;
  insert_text?: string;
  old_str?: string;
  view_range?: number[];
}

/** Resolve a tool-supplied path against one or more allowed roots and refuse
 *  anything that escapes all of them (directory-traversal / absolute-path guard).
 *  Relative paths are resolved against the primary root (`roots[0]`). */
export function resolveInCwd(p: string, roots: string | string[]): string {
  const rootList = (Array.isArray(roots) ? roots : [roots]).map((r) => resolve(r));
  const target = resolve(isAbsolute(p) ? p : join(rootList[0], p));
  if (!rootList.some((root) => target === root || target.startsWith(root + sep))) {
    throw new Error(`path "${p}" escapes the working directory`);
  }
  return target;
}

/**
 * Pure handler for the str_replace-based text editor tool: view / create /
 * str_replace / insert, sandboxed to `cwd`. Mirrors Anthropic's tool contract
 * (1-indexed line numbers, exactly-one-match str_replace, `insert_line` 0 =
 * top-of-file) so it backs both the provider-defined anthropic tool and the
 * generic fallback. Returns a human-readable string (errors as `Error: …`).
 */
export function textEdit(input: TextEditInput, roots: string | string[]): string {
  let target: string;
  try {
    target = resolveInCwd(input.path, roots);
  } catch (e) {
    return `Error: ${(e as Error).message}`;
  }

  switch (input.command) {
    case "view": {
      if (!existsSync(target)) return "Error: File not found";
      if (statSync(target).isDirectory()) {
        const entries = readdirSync(target, { withFileTypes: true })
          .filter((e) => !e.name.startsWith("."))
          .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
          .sort();
        return entries.length ? entries.join("\n") : "(empty directory)";
      }
      const lines = readFileSync(target, "utf-8").split("\n");
      let start = 1;
      let end = lines.length;
      if (Array.isArray(input.view_range) && input.view_range.length === 2) {
        start = Math.max(1, input.view_range[0]);
        end = input.view_range[1] === -1 ? lines.length : input.view_range[1];
      }
      const out = lines
        .slice(start - 1, end)
        .map((l, i) => `${start + i}: ${l}`)
        .join("\n");
      return out.length > FILE_VIEW_MAX_CHARS
        ? out.slice(0, FILE_VIEW_MAX_CHARS) + "\n\n[... output truncated ...]"
        : out;
    }

    case "create": {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, input.file_text ?? "");
      return `Successfully created ${input.path}`;
    }

    case "str_replace": {
      if (!existsSync(target)) return "Error: File not found";
      const content = readFileSync(target, "utf-8");
      const old = input.old_str ?? "";
      const count = old ? content.split(old).length - 1 : 0;
      if (count === 0)
        return "Error: No match found for replacement. Please check your text and try again.";
      if (count > 1)
        return `Error: Found ${count} matches for replacement text. Please provide more context to make a unique match.`;
      writeFileSync(target, content.replace(old, input.new_str ?? ""));
      return "Successfully replaced text at exactly one location.";
    }

    case "insert": {
      if (!existsSync(target)) return "Error: File not found";
      const lines = readFileSync(target, "utf-8").split("\n");
      const at = input.insert_line ?? 0;
      if (at < 0 || at > lines.length)
        return `Error: insert_line ${at} is out of range (0-${lines.length})`;
      lines.splice(at, 0, input.insert_text ?? "");
      writeFileSync(target, lines.join("\n"));
      return `Successfully inserted text after line ${at}`;
    }

    default:
      return `Error: unknown command "${(input as { command?: string }).command}"`;
  }
}
