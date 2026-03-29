import { useIngestion } from "@/stores/useIngestion";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Layers,
  FileCode,
  Braces,
  Globe,
  Database,
  Beaker,
  ChevronDown,
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
 * Parse GitHub URL to extract owner/repo and strip owner/repo from file paths
 */
function parseGitHubUrl(url: string | undefined | null, branch: string): GitHubInfo | null {
  if (!url) return null;

  try {
    // Handle both SSH and HTTPS formats
    // SSH: git@github.com:owner/repo.git
    // HTTPS: https://github.com/owner/repo or https://github.com/owner/repo.git
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

  // Find where the repo name is in the path
  const repoIndex = parts.findIndex(part => part === repo);

  if (repoIndex >= 0 && repoIndex < parts.length - 1) {
    // Return everything after the repo name
    return parts.slice(repoIndex + 1).join("/");
  }

  // Fallback: assume first two parts are owner/repo
  if (parts.length > 2) {
    return parts.slice(2).join("/");
  }

  return filePath;
}

/**
 * Get icon for node type
 */
function getNodeIcon(nodeType: ProvenanceCodeEntity["nodeType"]) {
  const iconProps = { className: "w-3 h-3 mr-1.5" };

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
 * Code entity node component
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

  return (
    <div className="pl-4 border-l border-border/30 ml-2">
      <a
        href={githubUrl || "#"}
        target="_blank"
        rel="noopener noreferrer"
        className={`flex items-center py-1 text-xs hover:text-primary transition-colors group ${
          !githubUrl ? "cursor-not-allowed opacity-50" : ""
        }`}
        onClick={(e) => {
          if (!githubUrl) e.preventDefault();
        }}
        title={
          githubUrl
            ? `Open ${entity.name} in GitHub`
            : "GitHub URL not available"
        }
      >
        {getNodeIcon(entity.nodeType)}
        <span className="flex-1">
          {entity.name}
          <span className="text-muted-foreground ml-1">(line {entity.start})</span>
        </span>
        {githubUrl && (
          <ExternalLink className="w-3 h-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </a>
    </div>
  );
}

/**
 * File node component
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
      <Collapsible defaultOpen>
        <div className="flex items-center py-1">
          <CollapsibleTrigger className="flex items-center flex-1 text-sm hover:text-primary transition-colors group">
            <ChevronDown className="w-3 h-3 mr-1 transition-transform [[data-state=closed]>&]:rotate-[-90deg]" />
            <FileCode
              className="w-3 h-3 mr-1.5"
              style={{ color: "#6b7280" }}
            />
            <span className="flex-1 text-left">{file.name}</span>
          </CollapsibleTrigger>
          {githubUrl && (
            <a
              href={githubUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="opacity-0 group-hover:opacity-100 transition-opacity"
              title={`Open ${file.name} in GitHub`}
            >
              <ExternalLink className="w-3 h-3 text-muted-foreground hover:text-primary" />
            </a>
          )}
        </div>
        <CollapsibleContent>
          {file.codeEntities.map((entity) => (
            <CodeEntityNode
              key={entity.refId}
              entity={entity}
              githubInfo={githubInfo}
            />
          ))}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

/**
 * Concept node component
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
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center w-full py-1.5 text-sm font-medium hover:text-primary transition-colors group">
          <ChevronDown className="w-4 h-4 mr-1.5 transition-transform [[data-state=closed]>&]:rotate-[-90deg]" />
          <Layers className="w-4 h-4 mr-2" style={{ color: "#64748b" }} />
          <span className="flex-1 text-left">{concept.name}</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1">
          {concept.description && (
            <div className="pl-6 pb-2 text-xs text-muted-foreground/80">
              {concept.description}
            </div>
          )}
          {concept.files.map((file) => (
            <FileNode key={file.refId} file={file} githubInfo={githubInfo} />
          ))}
        </CollapsibleContent>
      </Collapsible>
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
  const repoUrl = useIngestion((s) => s.repoUrl);
  // Use the first repo URL if comma-separated
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
