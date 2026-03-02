import { XMLParser } from "fast-xml-parser";

export type ParsedBounds = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

export type AndroidTreeElement = {
  index: number;
  className?: string;
  text?: string;
  resourceId?: string;
  accessibilityId?: string;
  clickable: boolean;
  boundsRaw?: string;
  bounds?: ParsedBounds;
  selectors: {
    resourceId?: string;
    accessibilityId?: string;
    text?: string;
    xpath?: string;
  };
};

export type AccessibilityTreeResult = {
  elements: AndroidTreeElement[];
};

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "",
  parseTagValue: false,
  parseAttributeValue: false,
  trimValues: true,
});

const ignoredNodeKeys = new Set([
  "index",
  "package",
  "class",
  "text",
  "resource-id",
  "content-desc",
  "checkable",
  "checked",
  "clickable",
  "enabled",
  "focusable",
  "focused",
  "long-clickable",
  "password",
  "scrollable",
  "selected",
  "bounds",
  "displayed",
  "a11y-important",
  "screen-reader-focusable",
  "drawing-order",
  "showing-hint",
  "text-entry-key",
  "dismissable",
  "a11y-focused",
  "heading",
  "live-region",
  "context-clickable",
  "content-invalid",
  "window-id",
]);

function toArray<T>(value: T | T[] | undefined): T[] {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function collectChildNodes(
  node: Record<string, unknown>,
): Record<string, unknown>[] {
  const children: Record<string, unknown>[] = [];

  for (const key of Object.keys(node)) {
    if (ignoredNodeKeys.has(key)) {
      continue;
    }

    const value = node[key];
    if (Array.isArray(value)) {
      for (const child of value) {
        if (child && typeof child === "object") {
          children.push(child as Record<string, unknown>);
        }
      }
      continue;
    }

    if (value && typeof value === "object") {
      children.push(value as Record<string, unknown>);
    }
  }

  return children;
}

function parseBounds(raw?: string): ParsedBounds | undefined {
  if (!raw) {
    return undefined;
  }

  const match = raw.match(/^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$/);
  if (!match) {
    return undefined;
  }

  return {
    left: Number(match[1]),
    top: Number(match[2]),
    right: Number(match[3]),
    bottom: Number(match[4]),
  };
}

function buildRelativeXpath(node: Record<string, unknown>): string {
  const className = typeof node.class === "string" ? node.class : undefined;
  const resourceId = typeof node["resource-id"] === "string" ? node["resource-id"] : undefined;
  const text = typeof node.text === "string" ? node.text : undefined;
  const contentDesc = typeof node["content-desc"] === "string" ? node["content-desc"] : undefined;

  if (resourceId) {
    return `//*[@resource-id="${resourceId}"]`;
  }

  if (contentDesc) {
    return `//*[@content-desc="${contentDesc}"]`;
  }

  if (text) {
    return `//*[@text="${text}"]`;
  }

  if (className) {
    return `//${className}`;
  }

  return "//node";
}

function walkNodes(
  node: Record<string, unknown>,
  collector: AndroidTreeElement[]
): void {
  const isClickable = String(node.clickable || "false") === "true";
  const rawText = typeof node.text === "string" ? node.text : undefined;
  const text = rawText && rawText.trim() !== "" ? rawText : undefined;
  const rawContentDesc =
    typeof node["content-desc"] === "string" ? node["content-desc"] : undefined;
  const contentDesc =
    rawContentDesc && rawContentDesc.trim() !== "" ? rawContentDesc : undefined;
  const isA11yImportant = String(node["a11y-important"] || "false") === "true";

  if (isClickable || text || contentDesc || isA11yImportant) {
    const resourceId =
      typeof node["resource-id"] === "string" && node["resource-id"] !== ""
        ? node["resource-id"]
        : undefined;

    const accessibilityId = contentDesc;

    const boundsRaw = typeof node.bounds === "string" ? node.bounds : undefined;

    collector.push({
      index: collector.length,
      className: typeof node.class === "string" ? node.class : undefined,
      text,
      resourceId,
      accessibilityId,
      clickable: isClickable,
      boundsRaw,
      bounds: parseBounds(boundsRaw),
      selectors: {
        resourceId,
        accessibilityId,
        text,
        xpath: buildRelativeXpath(node),
      },
    });
  }

  const children = collectChildNodes(node);
  for (const child of children) {
    walkNodes(child, collector);
  }
}

export function parseAccessibilityTree(xmlSource: string): AccessibilityTreeResult {
  const xmlObject = parser.parse(xmlSource) as {
    hierarchy?: Record<string, unknown>;
  };

  const roots = xmlObject.hierarchy
    ? collectChildNodes(xmlObject.hierarchy)
    : [];
  const elements: AndroidTreeElement[] = [];

  for (const root of roots) {
    walkNodes(root, elements);
  }

  return { elements };
}