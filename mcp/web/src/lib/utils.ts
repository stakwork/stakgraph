import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { GraphData } from "@/graph/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function resolveRepoUrl(
  data: GraphData | null,
  storedRepoUrl: string | null,
): string | null {
  if (data?.nodes) {
    const repoNodes = data.nodes.filter((n) => n.node_type === "Repository");
    if (repoNodes.length > 0) {
      return repoNodes
        .map((n) => {
          const sourceLink = n.properties.source_link as string | undefined;
          if (sourceLink) return sourceLink;
          return `https://github.com/${n.properties.name}`;
        })
        .join(",");
    }
  }

  return storedRepoUrl;
}
