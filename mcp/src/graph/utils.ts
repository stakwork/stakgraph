import { Node, Neo4jNode, ReturnNode, NodeType, toNum } from "./types.js";
import { Data_Bank } from "./neo4j.js";
import { simpleGit } from "simple-git";
import path from "path";
import fg from "fast-glob";
import { LANGUAGE_PACKAGE_FILES, Language } from "./types.js";

export function isTrue(value: string): boolean {
  return value === "true" || value === "1" || value === "True";
}

export const IS_TEST = isTrue(process.env.TEST_REF_ID as string);

export function rightLabel(node: Neo4jNode): string {
  let label = node.labels[0];
  if (label === Data_Bank) {
    label = node.labels[1] || "";
  }
  return label;
}

export function toReturnNode(node: Neo4jNode): ReturnNode {
  const properties = node.properties;
  const ref_id = IS_TEST ? "test_ref_id" : properties.ref_id || "";
  delete properties.ref_id;
  delete properties.text_embeddings;
  delete properties.embeddings;
  return {
    node_type: rightLabel(node) as NodeType,
    ref_id,
    properties,
  };
}

export function nameFileOnly(node: Neo4jNode): { name: string; file: string } {
  return {
    name: node.properties.name,
    file: node.properties.file,
  };
}

export function getNodeLabel(node: any) {
  if (!node.labels) {
    console.log("Node has no labels:", node);
    throw new Error("Node has no labels");
  }
  let label = rightLabel(node);
  const props = node.properties;
  let name = props.name;
  if (props.verb) {
    return `${label}: ${props.verb} ${name}`;
  } else {
    return `${label}: ${name}`;
  }
}

// Helper function to format node
export function formatNode(node: Neo4jNode): string {
  if (node && node.properties) {
    // Regular format for other nodes
    const ref_id = IS_TEST
      ? "test_ref_id"
      : node.ref_id || node.properties.ref_id || "N/A";
    return [
      `<snippet>`,
      `name: ${getNodeLabel(node)}`,
      `ref_id: ${ref_id}`,
      `file: ${node.properties.file || "Not specified"}`,
      `start: ${node.properties.start || "N/A"}, end: ${
        node.properties.end || "N/A"
      }`,
      node.properties.body ? "```\n" + node.properties.body + "\n```" : "",
      "</snippet>",
      "", // Empty line for spacing
    ].join("\n");
  }
  return "";
}

export function create_node_key(node: Node) {
  const { node_type, node_data } = node;
  const { name, file, start, verb } = node_data;
  const parts = [node_type, name, file, (start || 0).toString()];
  if (verb) parts.push(verb);
  const sanitized_parts = parts.map((part) => {
    return part
      .toLowerCase()
      .trim()
      .replace(/\s+/g, "")
      .replace(/[^a-zA-Z0-9]/g, "");
  });
  return sanitized_parts.join("-");
}

export function deser_node(record: any, key: string): Neo4jNode {
  const n: Neo4jNode = record.get(key);
  return clean_node(n);
}

export function deser_multi(record: any, key: string): Neo4jNode[] {
  const nodes: Neo4jNode[] = record.get(key);
  for (const n of nodes) {
    clean_node(n);
  }
  return nodes;
}

export function clean_node(n: Neo4jNode): Neo4jNode {
  if (n.properties.start) {
    n.properties.start = toNum(n.properties.start);
  }
  if (n.properties.end) {
    n.properties.end = toNum(n.properties.end);
  }
  if (n.properties.token_count) {
    n.properties.token_count = toNum(n.properties.token_count);
  }
  return n;
}

/**
 * @param language The programming language name (case insensitive)
 * @returns Array of file extensions including the dot (e.g. ['.js', '.jsx'])
 */
export function getExtensionsForLanguage(language: string): string[] {
  const lang = language.toLowerCase();

  const languageMap: Record<string, string[]> = {
    javascript: [".js", ".jsx", ".cjs", ".mjs"],
    typescript: [".ts", ".tsx", ".d.ts"],
    python: [".py", ".pyi", ".pyw"],
    ruby: [".rb", ".rake", ".gemspec"],
    go: [".go"],
    rust: [".rs"],
    java: [".java"],
    kotlin: [".kt", ".kts"],
    swift: [".swift"],
    c: [".c", ".h"],
    cpp: [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"],
    csharp: [".cs"],
    php: [".php"],
    html: [".html", ".htm"],
    css: [".css", ".scss", ".sass", ".less"],
    shell: [".sh", ".bash", ".zsh"],
    sql: [".sql"],
    markdown: [".md", ".markdown"],
    json: [".json"],
    yaml: [".yml", ".yaml"],
    xml: [".xml"],
    svelte: [".svelte"],
    vue: [".vue"],
    angular: [
      ".component.ts",
      ".service.ts",
      ".directive.ts",
      ".pipe.ts",
      ".module.ts",
    ],
  };

  return languageMap[lang] || [];
}

export async function cloneRepoToTmp(
  repoUrl: string,
  username?: string,
  pat?: string,
  commit?: string
): Promise<string> {
  const repoName =
    repoUrl
      .split("/")
      .pop()
      ?.replace(/\.git$/, "") || "repo";
  const cloneDir = path.join("/tmp", repoName + "-" + Date.now());

  let url = repoUrl;
  if (username && pat) {
    url = repoUrl.replace("https://", `https://${username}:${pat}@`);
  }

  await simpleGit().clone(url, cloneDir);

  if (commit) {
    await simpleGit(cloneDir).checkout(commit);
  }

  return cloneDir;
}
export async function detectLanguagesAndPkgFiles(
  repoDir: string
): Promise<{ language: Language; pkgFile: string }[]> {
  const detected: { language: Language; pkgFile: string }[] = [];
  for (const lang of Object.values(Language)) {
    const patterns = (LANGUAGE_PACKAGE_FILES[lang] || []).map((f) => `**/${f}`);
    const found = await fg(patterns, { cwd: repoDir, absolute: true });
    for (const pkgFile of found) {
      detected.push({ language: lang, pkgFile });
    }
  }
  return detected;
}
