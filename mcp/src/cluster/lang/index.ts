import typescript from './typescript.js';
import go from './go.js';
import python from './python.js';
import rust from './rust.js';
export type { LangConfig } from './types.js';
import type { LangConfig } from './types.js';

const configs: Record<string, LangConfig> = { typescript, go, python, rust };

function resolvedConfig(lang: LangConfig): {
  folders: Set<string>;
  filenames: Set<string>;
  isDynamic: (s: string) => boolean;
} {
  return {
    folders: new Set(lang.genericFolders),
    filenames: new Set(lang.genericFilenames),
    isDynamic: lang.isDynamicSegment,
  };
}

function isGeneric(segment: string, cfg: ReturnType<typeof resolvedConfig>): boolean {
  const lower = segment.toLowerCase();
  return cfg.folders.has(lower) || cfg.filenames.has(lower) || cfg.isDynamic(segment) || segment.length <= 1;
}

export function detectLang(files: string[]): LangConfig {
  const extCounts: Record<string, number> = {};
  for (const f of files) {
    const ext = f.split('.').pop()?.toLowerCase() || '';
    extCounts[ext] = (extCounts[ext] || 0) + 1;
  }
  const top = Object.entries(extCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '';
  if (['go'].includes(top)) return configs.go;
  if (['py'].includes(top)) return configs.python;
  if (['rs'].includes(top)) return configs.rust;
  return configs.typescript;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function stripSegment(raw: string): string {
  return raw.replace(/\.[^.]+$/, '').replace(/\.(test|spec)$/, '');
}

function bestSegment(file: string, cfg: ReturnType<typeof resolvedConfig>): string | null {
  const parts = file.split('/').filter(Boolean);
  for (let i = parts.length - 1; i >= 0; i--) {
    const segment = stripSegment(parts[i]);
    if (!isGeneric(segment, cfg)) return segment;
  }
  return null;
}

function bestParent(file: string, cfg: ReturnType<typeof resolvedConfig>): string | null {
  const parts = file.split('/').filter(Boolean);
  for (let i = parts.length - 2; i >= 0; i--) {
    const segment = stripSegment(parts[i]);
    if (!isGeneric(segment, cfg)) return segment;
  }
  return null;
}

export function generateLabel(
  files: string[],
  commNum: number,
  usedLabels: Set<string>,
  lang: LangConfig,
  fallbackPrefix = 'Cluster',
): string {
  const cfg = resolvedConfig(lang);
  const scores = new Map<string, number>();
  for (const file of files) {
    const seg = bestSegment(file, cfg);
    if (seg) {
      scores.set(seg, (scores.get(seg) || 0) + 1);
      continue;
    }
  }

  if (scores.size === 0) return `${fallbackPrefix}_${commNum}`;

  const candidates = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  for (const [segment] of candidates) {
    const label = capitalize(segment);
    if (!usedLabels.has(label)) {
      usedLabels.add(label);
      return label;
    }
  }

  const best = capitalize(candidates[0][0]);
  const parentCounts = new Map<string, number>();
  for (const file of files) {
    const parent = bestParent(file, cfg);
    if (parent) parentCounts.set(parent, (parentCounts.get(parent) || 0) + 1);
  }
  const parentLabel = parentCounts.size > 0
    ? capitalize([...parentCounts.entries()].sort((a, b) => b[1] - a[1])[0][0])
    : `${commNum}`;
  const disambiguated = `${best}-${parentLabel}`;
  usedLabels.add(disambiguated);
  return disambiguated;
}
