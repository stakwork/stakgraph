// Theme for the Canvas page — vendored 1:1 from
// /Users/evanfeenstra/code/evanf/system-diagram/demo/src/gateway.ts
// (the design's source-of-truth playground). Mirroring exactly so
// any further iteration there ports straight back here.
//
// Differences from the demo:
//   - No mock data — `Canvas.tsx` feeds the real
//     /_plugin/spend/by-agent-user response into `buildCanvas`.
//   - The `PROVIDER_DISPLAY` / `providerIcon` helpers (kept) are
//     used by Canvas.tsx to map the env-side provider id (`google`)
//     to a display label (`Gemini`) and the icon-lookup key.
//   - `platforms.ts` is gone — folded back into this file like the
//     demo does, since the four brand-icon path strings are the
//     entire payload.
//
// React-on-Preact note: `createElement` and `React` import from
// 'react' resolve through the vite alias to preact/compat (see
// vite.config.ts). preact/compat exports both, so the demo file
// runs unchanged.

import React, { createElement } from "react";
import type { CanvasTheme, SlotContext } from "system-canvas";
import { midnightTheme, resolveTheme } from "system-canvas";

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmtUSD(v: number): string {
  if (!v) return "$0.00";
  const digits = Math.abs(v) < 0.01 ? 6 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
}

// ---------------------------------------------------------------------------
// Custom body / footer renderers
// ---------------------------------------------------------------------------

/**
 * Render the gateway's hub-and-spoke network glyph + title as a single
 * `body` slot. The icon sits centered horizontally near the top of the
 * body region; the "Agent Gateway" title sits centered below it.
 * Authored in a 24-unit source box and scaled to `iconSize`.
 */
function renderGatewayBody(ctx: SlotContext): React.ReactNode {
  const { region, theme } = ctx;
  const iconSize = 30;
  const titleFs = 16;
  const gap = 12;
  const blockHeight = iconSize + gap + titleFs;
  const blockTop = region.y + (region.height - blockHeight) / 2;
  const iconCx = region.x + region.width / 2;
  const iconCy = blockTop + iconSize / 2;
  const titleY = blockTop + iconSize + gap + titleFs * 0.82;

  const src = 24;
  const scale = iconSize / src;
  const sx = (u: number) => iconCx + (u - src / 2) * scale;
  const sy = (u: number) => iconCy + (u - src / 2) * scale;

  const color = theme.node.labelColor;

  const hubCenter = { x: 12, y: 12 };
  const hubR = 3;
  const sqrt3over2 = Math.sqrt(3) / 2;
  const hexVerts: Array<[number, number]> = [
    [hubCenter.x, hubCenter.y - hubR],
    [hubCenter.x + hubR * sqrt3over2, hubCenter.y - hubR / 2],
    [hubCenter.x + hubR * sqrt3over2, hubCenter.y + hubR / 2],
    [hubCenter.x, hubCenter.y + hubR],
    [hubCenter.x - hubR * sqrt3over2, hubCenter.y + hubR / 2],
    [hubCenter.x - hubR * sqrt3over2, hubCenter.y - hubR / 2],
  ];
  const hexPath =
    "M " +
    hexVerts.map(([vx, vy]) => `${sx(vx)} ${sy(vy)}`).join(" L ") +
    " Z";

  const periR = 1.6;
  const peripherals = [
    { x: 6, y: 7 },
    { x: 18, y: 7 },
    { x: 12, y: 19.5 },
  ];

  const spokes = peripherals.map((p) => {
    const dx = p.x - hubCenter.x;
    const dy = p.y - hubCenter.y;
    const len = Math.hypot(dx, dy);
    const ux = dx / len;
    const uy = dy / len;
    return {
      x1: hubCenter.x + ux * hubR,
      y1: hubCenter.y + uy * hubR,
      x2: p.x - ux * periR,
      y2: p.y - uy * periR,
    };
  });

  const strokeWidth = 1.6;
  const ringR = 12;

  return createElement(
    "g",
    { pointerEvents: "none" },
    createElement("circle", {
      key: "ring",
      cx: sx(hubCenter.x),
      cy: sy(hubCenter.y),
      r: ringR * scale,
      fill: "none",
      stroke: color,
      strokeWidth,
      opacity: 0.85,
    }),
    ...spokes.map((s, i) =>
      createElement("line", {
        key: `spoke-${i}`,
        x1: sx(s.x1),
        y1: sy(s.y1),
        x2: sx(s.x2),
        y2: sy(s.y2),
        stroke: color,
        strokeWidth,
        strokeLinecap: "round",
        opacity: 0.85,
      }),
    ),
    createElement("path", {
      key: "hub",
      d: hexPath,
      fill: "none",
      stroke: color,
      strokeWidth,
      strokeLinejoin: "round",
      opacity: 0.95,
    }),
    ...peripherals.map((p, i) =>
      createElement("circle", {
        key: `node-${i}`,
        cx: sx(p.x),
        cy: sy(p.y),
        r: periR * scale,
        fill: "none",
        stroke: color,
        strokeWidth,
        opacity: 0.9,
      }),
    ),
    createElement(
      "text",
      {
        key: "title",
        x: iconCx,
        y: titleY,
        textAnchor: "middle",
        fill: color,
        fontSize: titleFs,
        fontFamily: theme.node.labelFont ?? theme.node.fontFamily,
        fontWeight: 600,
      },
      "Agent Gateway",
    ),
  );
}

/**
 * Two-piece footer: left label + right value, vertically centered in
 * the footer region. Used by agent/user cards to carry both a runs
 * count and a cost on the same row.
 */
function renderSplitFooter(
  ctx: SlotContext,
  left: string,
  right: string,
  opts: { fontSize?: number; color?: string } = {},
): React.ReactNode {
  const { region, theme } = ctx;
  const fs = opts.fontSize ?? 11;
  const color = opts.color ?? "#9ca3af";
  const y = region.y + region.height / 2 + fs * 0.36;
  const font = theme.node.fontFamily;
  const common = {
    y,
    fill: color,
    fontSize: fs,
    fontFamily: font,
    fontWeight: 500,
    pointerEvents: "none" as const,
  };
  return createElement(
    "g",
    { pointerEvents: "none" },
    left &&
      createElement(
        "text",
        { ...common, x: region.x, textAnchor: "start" },
        left,
      ),
    right &&
      createElement(
        "text",
        { ...common, x: region.x + region.width, textAnchor: "end" },
        right,
      ),
  );
}

// ---------------------------------------------------------------------------
// Icon path data
// ---------------------------------------------------------------------------

// Brand silhouettes from simple-icons (24-unit viewBox, fill mode).
const ANTHROPIC_PATHS = [
  "M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z",
];
const OPENAI_PATHS = [
  "M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z",
];
const GEMINI_PATHS = [
  "M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81",
];
const OPENROUTER_PATHS = [
  // Hand-rolled chain-link line glyph in 16-unit space, stroke mode.
  "M 7 9 L 5 11 A 2.5 2.5 0 0 1 1.5 7.5 L 4 5",
  "M 9 7 L 11 5 A 2.5 2.5 0 0 1 14.5 8.5 L 12 11",
  "M 6 10 L 10 6",
];

// Person + bot glyphs in a 24-unit source box. The `l 0.001 0` segments
// are degenerate dots — at stroke-linecap='round' they render as filled
// discs of diameter = stroke-width.
const PERSON_PATHS = [
  "M 8.5 8 A 3.5 3.5 0 1 0 15.5 8 A 3.5 3.5 0 1 0 8.5 8",
  "M 5 20 c 0 -3.5 3.13 -6 7 -6 s 7 2.5 7 6",
];
const BOT_PATHS = [
  // Antenna stub + tip dot
  "M 12 4 v 2",
  "M 12 3.5 l 0.001 0",
  // Head/body rounded rect (y=7..20)
  "M 7 7 H 17 A 2.5 2.5 0 0 1 19.5 9.5 V 17.5 A 2.5 2.5 0 0 1 17 20 H 7 A 2.5 2.5 0 0 1 4.5 17.5 V 9.5 A 2.5 2.5 0 0 1 7 7 Z",
  // Eyes
  "M 9 11.5 l 0.001 0",
  "M 15 11.5 l 0.001 0",
  // Mouth
  "M 10 15.5 h 4",
];

const PROVIDER_ICON_META: Record<
  string,
  { mode: "fill" | "stroke"; viewBox: 16 | 24 }
> = {
  anthropic: { mode: "fill", viewBox: 24 },
  openai: { mode: "fill", viewBox: 24 },
  gemini: { mode: "fill", viewBox: 24 },
  openrouter: { mode: "stroke", viewBox: 16 },
};

// ---------------------------------------------------------------------------
// Provider display table (label + icon key + brand color)
//
// Maps the env-side id (which can be `google` even though the brand is
// "Gemini") to the user-facing display label, the icon lookup key in
// `theme.icons`, and the per-card stroke color. Canvas.tsx reads this
// when building provider nodes.
// ---------------------------------------------------------------------------

export const PROVIDER_DISPLAY: Record<
  string,
  { label: string; icon: string; color: string }
> = {
  anthropic: { label: "Anthropic", icon: "anthropic", color: "#d97706" },
  openai: { label: "OpenAI", icon: "openai", color: "#10a37f" },
  google: { label: "Gemini", icon: "gemini", color: "#4285f4" },
  openrouter: { label: "OpenRouter", icon: "openrouter", color: "#a855f7" },
};

export function providerIcon(name: string): string {
  return PROVIDER_DISPLAY[name]?.icon ?? "anthropic";
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

export const canvasTheme: CanvasTheme = resolveTheme(
  {
    name: "gateway",
    icons: {
      bot: BOT_PATHS,
      person: PERSON_PATHS,
      anthropic: ANTHROPIC_PATHS,
      openai: OPENAI_PATHS,
      gemini: GEMINI_PATHS,
      openrouter: OPENROUTER_PATHS,
    },
    categories: {
      // ─── agent ───────────────────────────────────────────────────
      // Small per-(agent × user) card. Bot icon top-left, agent name
      // as the header, custom split footer carrying runs + cost.
      agent: {
        defaultWidth: 160,
        defaultHeight: 52,
        fill: "rgba(52, 211, 153, 0.08)",
        stroke: "#34d399",
        cornerRadius: 6,
        type: "text",
        slots: {
          leftEdge: { kind: "color", extent: "full" },
          topLeft: { kind: "icon", name: "bot", viewBox: 24, size: 14 },
          header: {
            kind: "text",
            value: (ctx: SlotContext) =>
              (ctx.node.customData?.name as string) ?? "",
            fontSize: 13,
            uppercase: false,
            useLabelFont: false,
            color: (ctx: SlotContext) => ctx.theme.node.labelColor,
          },
          footer: {
            kind: "custom",
            render: (ctx: SlotContext) =>
              renderSplitFooter(
                ctx,
                `${(ctx.node.customData?.calls as number) ?? 0} runs`,
                fmtUSD((ctx.node.customData?.cost as number) ?? 0),
              ),
          },
        },
      },

      // ─── user ────────────────────────────────────────────────────
      // Same recipe as agent, slightly bigger. Header = username,
      // footer = "N runs" + total spend.
      user: {
        defaultWidth: 180,
        defaultHeight: 64,
        fill: "rgba(34, 211, 238, 0.10)",
        stroke: "#22d3ee",
        cornerRadius: 8,
        type: "text",
        slots: {
          leftEdge: { kind: "color", extent: "full" },
          topLeft: { kind: "icon", name: "person", viewBox: 24 },
          header: {
            kind: "text",
            value: (ctx: SlotContext) =>
              (ctx.node.customData?.name as string) ?? "",
            fontSize: 13,
            uppercase: false,
            useLabelFont: false,
            color: (ctx: SlotContext) => ctx.theme.node.labelColor,
          },
          footer: {
            kind: "custom",
            render: (ctx: SlotContext) =>
              renderSplitFooter(
                ctx,
                `${(ctx.node.customData?.requestCount as number) ?? 0} runs`,
                fmtUSD((ctx.node.customData?.totalCost as number) ?? 0),
              ),
          },
        },
      },

      // ─── gateway ─────────────────────────────────────────────────
      // Singleton hub. Custom body renders the hex+spokes glyph and
      // the "Agent Gateway" title together; footer carries the
      // swarm-wide total spend.
      gateway: {
        defaultWidth: 220,
        defaultHeight: 100,
        fill: "rgba(167, 139, 250, 0.10)",
        stroke: "#a78bfa",
        cornerRadius: 10,
        type: "text",
        slots: {
          topEdge: { kind: "color", extent: "full" },
          body: { kind: "custom", render: renderGatewayBody },
          footer: {
            kind: "text",
            value: (ctx: SlotContext) =>
              `${fmtUSD((ctx.node.customData?.totalCost as number) ?? 0)} swarm total`,
            fontSize: 11,
            align: "center",
            color: "#9ca3af",
          },
        },
      },

      // ─── provider ────────────────────────────────────────────────
      // Per-node brand color (set via `node.color` in Canvas.tsx)
      // drives the stroke + the rightEdge color stripe. Icon picked
      // per-node from `customData.icon` with per-provider mode +
      // viewBox dispatched off PROVIDER_ICON_META.
      provider: {
        defaultWidth: 180,
        defaultHeight: 64,
        fill: "rgba(251, 146, 60, 0.08)",
        stroke: "#fb923c",
        cornerRadius: 8,
        type: "text",
        slots: {
          rightEdge: { kind: "color", extent: "full" },
          topLeft: {
            kind: "icon",
            name: (ctx: SlotContext) =>
              (ctx.node.customData?.icon as string) ?? "anthropic",
            mode: (ctx: SlotContext) =>
              PROVIDER_ICON_META[ctx.node.customData?.icon as string]?.mode ??
              "fill",
            viewBox: (ctx: SlotContext) =>
              PROVIDER_ICON_META[ctx.node.customData?.icon as string]
                ?.viewBox ?? 24,
          },
          header: {
            kind: "text",
            value: (ctx: SlotContext) =>
              (ctx.node.customData?.name as string) ?? "",
            fontSize: 13,
            uppercase: false,
            useLabelFont: false,
            color: (ctx: SlotContext) => ctx.theme.node.labelColor,
          },
        },
      },
    },
  },
  midnightTheme,
);
