import { Node, Neo4jNode, ReturnNode, NodeType, toNum } from "./types.js";
import { Data_Bank } from "./neo4j.js";
import { simpleGit } from "simple-git";
import path from "path";
import fg from "fast-glob";
import fs from "fs/promises";
import {
  LANGUAGE_PACKAGE_FILES,
  Language,
  LANGUAGE_ENV_REGEX,
  EXTENSIONS,
  IGNORE_DIRECTORIES,
  IGNORE_FILES,
} from "./types.js";

export function isTrue(value: string): boolean {
  return value === "true" || value === "1" || value === "True";
}

export const IS_TEST = isTrue(process.env.TEST_REF_ID as string);

export function rightLabel(node: Neo4jNode): NodeType {
  let label = node.labels[0];
  if (label === Data_Bank) {
    label = node.labels[1] || "";
  }
  return label as NodeType;
}

export function toReturnNode(node: Neo4jNode): ReturnNode {
  const properties = node.properties;
  const ref_id = IS_TEST ? "test_ref_id" : properties.ref_id || "";
  delete properties.ref_id;
  delete properties.text_embeddings;
  delete properties.embeddings;
  return {
    node_type: rightLabel(node),
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

export function getNodeSummaryLabel(node: Neo4jNode) {
  if (!node.labels) {
    console.log("Node has no labels:", node);
    throw new Error("Node has no labels");
  }
  let label = rightLabel(node);

  // entire body
  if (label === "Import" || label === "Datamodel" || label === "Request") {
    return `${label}: \n${node.properties.body}`;
  }
  // first 5 lines of body
  if (label === "Function" || label == "Var" || label === "Endpoint") {
    const lines =
      node.properties.start != node.properties.end
        ? `lines ${node.properties.start} - ${node.properties.end}`
        : `line ${node.properties.start}`;
    let lab = `${label}: ${node.properties.name} (${lines})`;
    const bod = node.properties.body?.split("\n").slice(0, 10).join("\n");
    if (bod) {
      lab += `\n${bod}`;
    }
    return lab;
  }

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
  if (n.properties.ref_id) {
    n.ref_id = n.properties.ref_id;
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

export async function isBinaryFile(filePath: string): Promise<boolean> {
  try {
    const buf = await fs.readFile(filePath, { encoding: null });
    for (let i = 0; i < Math.min(buf.length, 8000); i++) {
      if (buf[i] == 0) return true;
    }
    return false;
  } catch {
    return true;
  }
}

export function getLanguageByExtension(ext: string): Language | undefined {
  ext = ext.replace(/^\./, "").toLowerCase();
  for (const [lang, exts] of Object.entries(EXTENSIONS)) {
    if (exts.map((e) => e.replace(/^\./, "")).includes(ext)) {
      return lang as Language;
    }
  }
  return undefined;
}

export async function extractEnvVarsFromRepo(
  repoDir: string
): Promise<Record<string, Set<string>>> {
  const envVarsByFile: Record<string, Set<string>> = {};

  const files = await fg("**/*.*", {
    cwd: repoDir,
    absolute: true,
    dot: true,
    ignore: [...IGNORE_DIRECTORIES, ...IGNORE_FILES],
  });

  for (const file of files) {
    if (await isBinaryFile(file)) continue;

    let body: string;

    try {
      body = await fs.readFile(file, "utf8");
    } catch (error) {
      console.error(`Error reading file ${file}:`, error);
      continue;
    }

    const ext = file.split(".").pop()?.toLowerCase() || "";
    const language = getLanguageByExtension(ext);

    if (!language) continue;

    const regex = LANGUAGE_ENV_REGEX[language];
    if (!regex) continue;

    const matches = [...body.matchAll(regex)].map((m) => m[1]);
    if (matches.length > 0) {
      if (!envVarsByFile[file]) envVarsByFile[file] = new Set();
      matches.forEach((v) => envVarsByFile[file].add(v));
    }
  }

  return envVarsByFile;
}

export async function findDockerComposeFiles(
  repoDir: string
): Promise<string[]> {
  const patterns = ["**/docker-compose.yml", "**/docker-compose.yaml"];
  const found = await fg(patterns, { cwd: repoDir, absolute: true });
  return found;
}
