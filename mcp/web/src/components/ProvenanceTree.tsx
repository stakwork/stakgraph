import { useMemo } from "react";
import { useIngestion } from "@/stores/useIngestion";
import { useGraphData } from "@/stores/useGraphData";
import {
  Layers,
  FileCode,
  Braces,
  Globe,
  Database,
  Beaker,
  ExternalLink,
} from "lucide-react";

/**
 * Code entity in provenance response
 */
export interface ProvenanceCodeEntity {
  refId: string;
  name: string;
  nodeType:
    | "Function"
    | "Page"
    | "Endpoint"
    | "Datamodel"
    | "UnitTest"
    | "IntegrationTest"
    | "E2etest";
  file: string;
  start: number;
  end: number;
}

/**
 * File with code entities in provenance response
 */
export interface ProvenanceFile {
  refId: string;
  name: string;
  path: string;
  codeEntities: ProvenanceCodeEntity[];
}

/**
 * Concept with files in provenance response
 */
export interface ProvenanceConcept {
  refId: string;
  name: string;
  description?: string;
  files: ProvenanceFile[];
}

/**
 * Response from /gitree/provenance endpoint
 */
export interface ProvenanceData {
  concepts: ProvenanceConcept[];
}

interface GitHubInfo {
  owner: string;
  repo: string;
  branch: string;
}

/**
 * Parse GitHub URL to extract owner/repo
 */
function parseGitHubUrl(url: string | undefined | null, branch: string): GitHubInfo | null {
  if (!url) return null;

  try {
    let owner: string;
    let repo: string;

    if (url.startsWith("git@github.com:")) {
      const parts = url.replace("git@github.com:", "").replace(".git", "").split("/");
      [owner, repo] = parts;
    } else {
      const urlObj = new URL(url);
      const pathParts = urlObj.pathname.replace(/^\//, "").replace(/\.git$/, "").split("/");
      [owner, repo] = pathParts;
    }

    return { owner, repo, branch };
  } catch {
    return null;
  }
}

/**
 * Strip owner/repo from file path (e.g., "stakwork/hive/src/file.ts" -> "src/file.ts")
 */
function stripOwnerRepoFromPath(filePath: string, repo: string): string {
  const parts = filePath.split("/");
  const repoIndex = parts.findIndex(part => part === repo);

  if (repoIndex >= 0 && repoIndex < parts.length - 1) {
    return parts.slice(repoIndex + 1).join("/");
  }

  if (parts.length > 2) {
    return parts.slice(2).join("/");
  }

  return filePath;
}

/**
 * Get icon for node type
 */
function getNodeIcon(nodeType: ProvenanceCodeEntity["nodeType"]) {
  const iconProps = { className: "w-3 h-3 mr-1.5 shrink-0" };

  switch (nodeType) {
    case "Function":
      return <Braces {...iconProps} style={{ color: "#8b5cf6" }} />;
    case "Page":
      return <Globe {...iconProps} style={{ color: "#10b981" }} />;
    case "Endpoint":
      return <Globe {...iconProps} style={{ color: "#f59e0b" }} />;
    case "Datamodel":
      return <Database {...iconProps} style={{ color: "#06b6d4" }} />;
    case "UnitTest":
    case "IntegrationTest":
    case "E2etest":
      return <Beaker {...iconProps} style={{ color: "#ec4899" }} />;
    default:
      return <Braces {...iconProps} style={{ color: "#8b5cf6" }} />;
  }
}

/**
 * Code entity node - links to specific line on GitHub
 */
function CodeEntityNode({
  entity,
  githubInfo,
}: {
  entity: ProvenanceCodeEntity;
  githubInfo: GitHubInfo | null;
}) {
  const cleanPath = githubInfo ? stripOwnerRepoFromPath(entity.file, githubInfo.repo) : entity.file;
  const githubUrl = githubInfo
    ? `https://github.com/${githubInfo.owner}/${githubInfo.repo}/tree/${githubInfo.branch}/${cleanPath}#L${entity.start}`
    : null;

  const inner = (
    <>
      {getNodeIcon(entity.nodeType)}
      <span className="flex-1 truncate">
        {entity.name}
        <span className="text-muted-foreground ml-1">(line {entity.start})</span>
      </span>
      {githubUrl && (
        <ExternalLink className="w-3 h-3 ml-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}
    </>
  );

  return (
    <div className="pl-4 border-l border-border/30 ml-2">
      {githubUrl ? (
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center py-1 text-xs cursor-pointer hover:text-primary transition-colors group"
          title={`Open ${entity.name} in GitHub`}
        >
          {inner}
        </a>
      ) : (
        <div className="flex items-center py-1 text-xs opacity-50">
          {inner}
        </div>
      )}
    </div>
  );
}

/**
 * File node - links to file on GitHub, with code entities listed below
 */
function FileNode({
  file,
  githubInfo,
}: {
  file: ProvenanceFile;
  githubInfo: GitHubInfo | null;
}) {
  const cleanPath = githubInfo ? stripOwnerRepoFromPath(file.path, githubInfo.repo) : file.path;
  const githubUrl = githubInfo
    ? `https://github.com/${githubInfo.owner}/${githubInfo.repo}/tree/${githubInfo.branch}/${cleanPath}`
    : null;

  return (
    <div className="pl-4 border-l border-border/30 ml-2">
      {githubUrl ? (
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center py-1 text-sm cursor-pointer hover:text-primary transition-colors group"
          title={`Open ${file.name} in GitHub`}
        >
          <FileCode className="w-3 h-3 mr-1.5 shrink-0" style={{ color: "#6b7280" }} />
          <span className="flex-1 text-left truncate">{file.name}</span>
          <ExternalLink className="w-3 h-3 ml-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
        </a>
      ) : (
        <div className="flex items-center py-1 text-sm">
          <FileCode className="w-3 h-3 mr-1.5 shrink-0" style={{ color: "#6b7280" }} />
          <span className="flex-1 text-left truncate">{file.name}</span>
        </div>
      )}
      {file.codeEntities.map((entity) => (
        <CodeEntityNode
          key={entity.refId}
          entity={entity}
          githubInfo={githubInfo}
        />
      ))}
    </div>
  );
}

/**
 * Concept node - heading with files and code entities below
 */
function ConceptNode({
  concept,
  githubInfo,
}: {
  concept: ProvenanceConcept;
  githubInfo: GitHubInfo | null;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center py-1.5 text-sm font-medium">
        <Layers className="w-4 h-4 mr-2 shrink-0" style={{ color: "#64748b" }} />
        <span className="flex-1 text-left">{concept.name}</span>
      </div>
      {concept.description && (
        <div className="pl-6 pb-2 text-xs text-muted-foreground/80">
          {concept.description}
        </div>
      )}
      {concept.files.map((file) => (
        <FileNode key={file.refId} file={file} githubInfo={githubInfo} />
      ))}
    </div>
  );
}

/**
 * ProvenanceTree component - displays knowledge sources used by AI
 */
export function ProvenanceTree({
  provenanceData,
}: {
  provenanceData: ProvenanceData;
}) {
  const storedRepoUrl = useIngestion((s) => s.repoUrl);
  const data = useGraphData((s) => s.data);

  // Derive repo URL from graph Repository nodes, fall back to ingestion store
  const repoUrl = useMemo(() => {
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
  }, [data, storedRepoUrl]);

  const firstUrl = repoUrl?.split(",")[0]?.trim() || null;
  const githubInfo = parseGitHubUrl(firstUrl, "main");

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold mb-3 flex items-center">
        <Layers className="w-4 h-4 mr-2" />
        Knowledge Sources
      </h3>
      {provenanceData.concepts.map((concept) => (
        <ConceptNode
          key={concept.refId}
          concept={concept}
          githubInfo={githubInfo}
        />
      ))}
    </div>
  );
}
