// Tiny inline-SVG icon set. We hand-roll these rather than pull in a
// library (lucide-preact, etc.) because:
//
//   - the icon count is small (provenance card + future phase-9 buttons),
//   - matching the dashboard's stroke weight + corner radius is easier
//     with hand-tuned SVGs than overriding a library's defaults,
//   - no extra dependency in the bundle.
//
// All icons render at 1em so they scale with the surrounding text.
// Stroke uses `currentColor` so the parent's `color` flows through —
// e.g. `text-dim` desaturates the icon along with the label.

import type { JSX } from "preact";

interface IconProps extends JSX.SVGAttributes<SVGSVGElement> {
  /** Pixel size; default 1em (scales with surrounding text). */
  size?: number | string;
}

function baseProps({ size = "1em", ...rest }: IconProps) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": 1.8,
    "stroke-linecap": "round" as const,
    "stroke-linejoin": "round" as const,
    "aria-hidden": true,
    ...rest,
  };
}

/** UserIcon — classic head + shoulders silhouette. */
export function UserIcon(props: IconProps) {
  return (
    <svg {...baseProps(props)}>
      <circle cx="12" cy="8" r="3.5" />
      <path d="M5 20c0-3.5 3.13-6 7-6s7 2.5 7 6" />
    </svg>
  );
}

/** ChevronLeftIcon — used by the sidebar collapse button. */
export function ChevronLeftIcon(props: IconProps) {
  return (
    <svg {...baseProps(props)}>
      <path d="M15 6l-6 6 6 6" />
    </svg>
  );
}

/** ChevronRightIcon — used by the topbar expand button when the
 *  sidebar is hidden. Mirror of ChevronLeftIcon. */
export function ChevronRightIcon(props: IconProps) {
  return (
    <svg {...baseProps(props)}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

/** BotIcon — friendly bot head (antenna, screen-face, side-ports). */
export function BotIcon(props: IconProps) {
  return (
    <svg {...baseProps(props)}>
      {/* Antenna */}
      <path d="M12 3v2" />
      <circle cx="12" cy="2.5" r="0.6" fill="currentColor" />
      {/* Head */}
      <rect x="4.5" y="6" width="15" height="12" rx="2.5" />
      {/* Eyes */}
      <circle cx="9" cy="12" r="1" fill="currentColor" />
      <circle cx="15" cy="12" r="1" fill="currentColor" />
      {/* Mouth */}
      <path d="M10 15.5h4" />
      {/* Side ports */}
      <path d="M3 11v2" />
      <path d="M21 11v2" />
    </svg>
  );
}
