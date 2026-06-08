import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { buildDriveClient, statusOf, describeDriveError } from "./_shared.js";

const EXAMPLE = `- id: doc
  type: gdrive/export-file
  config:
    fileId: "1AbCdEfGhIjKlMnOpQrStUvWxYz"
    accessToken: "{{ input.googleAccessToken }}"`;

// Default export format per Google-native ("application/vnd.google-apps.*")
// type. Anything not listed here falls back to text/plain; non-Google files
// are downloaded as-is via alt=media. Override with `exportMimeType`.
const EXPORT_MIME: Record<string, string> = {
  "application/vnd.google-apps.document": "text/markdown",
  "application/vnd.google-apps.spreadsheet": "text/csv",
  "application/vnd.google-apps.presentation": "text/plain",
  "application/vnd.google-apps.script": "application/vnd.google-apps.script+json",
};

const GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps";
const DEFAULT_NATIVE_EXPORT = "text/plain";

export default defineStep({
  type: "gdrive/export-file",
  description: `Fetch a Google Drive file and return its text content for LLM consumption. Google-native files (Docs → markdown, Sheets → CSV, Slides → text) are exported; other files are downloaded as-is. Auth: pass an OAuth \`accessToken\`, else falls back to Application Default Credentials (e.g. a service account via GOOGLE_APPLICATION_CREDENTIALS). Output: { content, truncated, file: { id, name, mimeType, modifiedTime, size, webViewLink } }.\n\n${EXAMPLE}`,
  input: z.object({
    fileId: z.string().min(1),
    accessToken: z.string().optional(),
    /** Override the export MIME type for Google-native files. Ignored for
     *  non-Google files (which are always downloaded as-is). */
    exportMimeType: z.string().optional(),
    /** Truncate content to this many characters (LLM token hygiene). */
    maxChars: z.number().int().positive().default(50000),
  }),
  output: z.object({
    content: z.string(),
    truncated: z.boolean(),
    file: z.object({
      id: z.string(),
      name: z.string(),
      mimeType: z.string(),
      modifiedTime: z.string().nullable(),
      size: z.number().nullable(),
      webViewLink: z.string().nullable(),
    }),
  }),
  async run(cfg, ctx: StepContext<VeinCapabilities>) {
    const { client, haveAuth } = await buildDriveClient(cfg.accessToken, ctx);
    const resource = `file "${cfg.fileId}"`;

    const { data: meta } = await client.files
      .get({
        fileId: cfg.fileId,
        fields: "id,name,mimeType,modifiedTime,size,webViewLink",
        supportsAllDrives: true,
      })
      .catch((err: unknown) => {
        throw describeDriveError(err, resource, haveAuth);
      });

    const mimeType = meta.mimeType ?? "application/octet-stream";
    const isNative = mimeType.startsWith(GOOGLE_NATIVE_PREFIX);

    let raw: unknown;
    if (isNative) {
      const exportMime =
        cfg.exportMimeType ?? EXPORT_MIME[mimeType] ?? DEFAULT_NATIVE_EXPORT;
      ({ data: raw } = await client.files
        .export({ fileId: cfg.fileId, mimeType: exportMime })
        .catch((err: unknown) => {
          if (statusOf(err) === 400) {
            throw new Error(
              `Google Drive cannot export "${meta.name ?? cfg.fileId}" (${mimeType}) as "${exportMime}". Set a supported \`exportMimeType\` for this file type.`,
            );
          }
          throw describeDriveError(err, resource, haveAuth);
        }));
    } else {
      ({ data: raw } = await client.files
        .get({ fileId: cfg.fileId, alt: "media", supportsAllDrives: true })
        .catch((err: unknown) => {
          throw describeDriveError(err, resource, haveAuth);
        }));
    }

    const full = coerceText(raw);
    const truncated = full.length > cfg.maxChars;
    const content = truncated
      ? `${full.slice(0, cfg.maxChars)}\n\n... [truncated ${full.length - cfg.maxChars} characters]`
      : full;

    return {
      content,
      truncated,
      file: {
        id: meta.id ?? cfg.fileId,
        name: meta.name ?? "unknown",
        mimeType,
        modifiedTime: meta.modifiedTime ?? null,
        size: meta.size != null ? Number(meta.size) : null,
        webViewLink: meta.webViewLink ?? null,
      },
    };
  },
});

/** Coerce a Drive export/download body into a string. Text exports come back
 *  as a string already; structured bodies are JSON-stringified as a fallback. */
function coerceText(data: unknown): string {
  if (typeof data === "string") return data;
  if (data == null) return "";
  if (data instanceof Uint8Array) return Buffer.from(data).toString("utf-8");
  return JSON.stringify(data);
}
