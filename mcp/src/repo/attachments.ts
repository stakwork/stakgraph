import { randomUUID } from "crypto";
import type { ImagePart, ModelMessage, TextPart } from "ai";
import {
  writeAttachment,
  readAttachment,
  appendAttachmentMeta,
  AttachmentMeta,
} from "./session.js";

/**
 * Scheme used for the lightweight placeholder stored in the conversation
 * JSONL. The real bytes live in the session-scoped attachment cache and are
 * rehydrated on each turn (see rehydrateMessages).
 */
export const ATTACHMENT_SCHEME = "attachment://";

const MAX_ATTACHMENTS = 8;
const MAX_BYTES = 15 * 1024 * 1024; // 15 MB per image
const DOWNLOAD_TIMEOUT_MS = 30_000;

const ATTACHMENTS_TAG_RE = /<attachments>([\s\S]*?)<\/attachments>/i;

export interface CachedAttachment {
  id: string;
  mediaType: string;
  bytes: Uint8Array;
}

/**
 * Extract `<attachments>url1,url2</attachments>` from a single text blob.
 * Returns the text with the tag removed and the list of URLs it contained.
 */
export function extractTagUrls(text: string): { cleanedText: string; urls: string[] } {
  const match = text.match(ATTACHMENTS_TAG_RE);
  if (!match) return { cleanedText: text, urls: [] };
  const urls = splitUrls(match[1]);
  const cleanedText = text.replace(ATTACHMENTS_TAG_RE, "").trim();
  return { cleanedText, urls };
}

function splitUrls(raw: string): string[] {
  return raw
    .split(",")
    .map((u) => u.trim())
    .filter((u) => /^https?:\/\//i.test(u));
}

/**
 * Resolve the image URLs for the CURRENT turn and strip the `<attachments>`
 * tag from the prompt text. The body field (bodyAttachments) takes
 * precedence; the tag is only a fallback and is only read from the LAST user
 * message so prior turns are never re-parsed.
 */
export function resolveCurrentTurnAttachments(
  prompt: string | ModelMessage[],
  bodyAttachments?: string[],
): { urls: string[]; cleanedPrompt: string | ModelMessage[] } {
  const bodyUrls = (bodyAttachments ?? []).filter((u) => /^https?:\/\//i.test(u));

  if (typeof prompt === "string") {
    const { cleanedText, urls: tagUrls } = extractTagUrls(prompt);
    const urls = dedupe(bodyUrls.length > 0 ? bodyUrls : tagUrls).slice(0, MAX_ATTACHMENTS);
    // Always strip the tag from the text, even when the body field wins.
    return { urls, cleanedPrompt: cleanedText };
  }

  // ModelMessage[]: only touch the last user message.
  const messages = [...prompt];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== "user") continue;
    const msg = messages[i];
    let tagUrls: string[] = [];
    if (typeof msg.content === "string") {
      const res = extractTagUrls(msg.content);
      tagUrls = res.urls;
      if (res.urls.length > 0) {
        messages[i] = { ...msg, content: res.cleanedText } as ModelMessage;
      }
    } else if (Array.isArray(msg.content)) {
      const parts = msg.content.map((part: any) => {
        if (part.type === "text" && typeof part.text === "string") {
          const res = extractTagUrls(part.text);
          tagUrls.push(...res.urls);
          return { ...part, text: res.cleanedText };
        }
        return part;
      });
      messages[i] = { ...msg, content: parts } as ModelMessage;
    }
    const urls = dedupe(bodyUrls.length > 0 ? bodyUrls : tagUrls).slice(0, MAX_ATTACHMENTS);
    return { urls, cleanedPrompt: messages };
  }

  return { urls: dedupe(bodyUrls).slice(0, MAX_ATTACHMENTS), cleanedPrompt: prompt };
}

function dedupe(urls: string[]): string[] {
  return Array.from(new Set(urls));
}

/** Best-effort text of the last user message in a message array. */
export function lastUserText(messages: ModelMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "user") continue;
    if (typeof msg.content === "string") return msg.content;
    if (Array.isArray(msg.content)) {
      const textPart = msg.content.find((p: any) => p.type === "text");
      return textPart && "text" in textPart ? (textPart as any).text : undefined;
    }
    return undefined;
  }
  return undefined;
}

/**
 * Download each URL once. When a sessionId is provided the bytes are written
 * to the session-scoped cache (and an audit line is appended); otherwise the
 * bytes are only held in memory for this single turn.
 * Non-image responses are skipped.
 */
export async function cacheAttachments(
  urls: string[],
  sessionId: string | undefined,
  signal?: AbortSignal,
): Promise<CachedAttachment[]> {
  const out: CachedAttachment[] = [];
  const meta: AttachmentMeta[] = [];

  for (const url of urls) {
    try {
      const { bytes, mediaType } = await downloadImage(url, signal);
      const id = randomUUID();
      if (sessionId) {
        writeAttachment(sessionId, id, bytes);
        meta.push({
          id,
          mediaType,
          bytes: bytes.byteLength,
          originalUrl: url,
          createdAt: new Date().toISOString(),
        });
      }
      out.push({ id, mediaType, bytes });
    } catch (err) {
      console.error(`[attachments] failed to fetch ${url}:`, err instanceof Error ? err.message : err);
    }
  }

  if (sessionId && meta.length > 0) {
    appendAttachmentMeta(sessionId, meta);
  }
  return out;
}

async function downloadImage(
  url: string,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; mediaType: string }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);
  const onParentAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", onParentAbort, { once: true });
  }
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const headerType = (res.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
    const mediaType = headerType.startsWith("image/") ? headerType : inferMediaType(url);
    if (!mediaType.startsWith("image/")) {
      throw new Error(`not an image (content-type: ${headerType || "unknown"})`);
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf.byteLength === 0) throw new Error("empty body");
    if (buf.byteLength > MAX_BYTES) {
      throw new Error(`too large (${buf.byteLength} bytes > ${MAX_BYTES})`);
    }
    return { bytes: buf, mediaType };
  } finally {
    clearTimeout(timeout);
    if (signal) signal.removeEventListener("abort", onParentAbort);
  }
}

function inferMediaType(url: string): string {
  const path = url.split("?")[0].toLowerCase();
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".jpg") || path.endsWith(".jpeg")) return "image/jpeg";
  if (path.endsWith(".gif")) return "image/gif";
  if (path.endsWith(".webp")) return "image/webp";
  return "";
}

/** Image parts (with real bytes) to send to the model this turn. */
export function buildImageParts(atts: CachedAttachment[]): ImagePart[] {
  return atts.map((a) => ({ type: "image", image: a.bytes, mediaType: a.mediaType }));
}

/** Lightweight placeholder parts persisted to the conversation JSONL. */
export function buildPlaceholderParts(atts: CachedAttachment[]): ImagePart[] {
  return atts.map((a) => ({
    type: "image",
    image: `${ATTACHMENT_SCHEME}${a.id}`,
    mediaType: a.mediaType,
  }));
}

/**
 * Replace `attachment://<id>` placeholder image parts in previously stored
 * messages with the cached bytes so the model can keep seeing the image on
 * follow-up turns. Missing bytes degrade to a harmless text note.
 */
export function rehydrateMessages(
  messages: ModelMessage[],
  sessionId: string,
): ModelMessage[] {
  let changed = false;
  const out = messages.map((msg) => {
    if (msg.role !== "user" || !Array.isArray(msg.content)) return msg;
    let msgChanged = false;
    const parts = msg.content.map((part: any) => {
      if (
        part?.type === "image" &&
        typeof part.image === "string" &&
        part.image.startsWith(ATTACHMENT_SCHEME)
      ) {
        const id = part.image.slice(ATTACHMENT_SCHEME.length);
        const bytes = readAttachment(sessionId, id);
        msgChanged = true;
        changed = true;
        if (bytes) {
          return { ...part, image: bytes } as ImagePart;
        }
        return { type: "text", text: "[attached image is no longer available]" } as TextPart;
      }
      return part;
    });
    return msgChanged ? ({ ...msg, content: parts } as ModelMessage) : msg;
  });
  return changed ? out : messages;
}
