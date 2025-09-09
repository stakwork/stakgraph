export interface Node {
  node_type: NodeType;
  node_data: NodeData;
}

export type BoltInt = number | { low: number; hight: number };

export interface Neo4jNode {
  identity?: BoltInt; // built-in on some queries
  properties: NodeData;
  labels: string[];
  ref_id?: string;
  score?: number;
}

export interface ReturnNode {
  node_type: NodeType;
  ref_id: string;
  properties: NodeData;
}

export interface NodeData {
  name: string;
  file: string;
  body: string;
  start: number;
  end: number;
  docs?: string;
  hash?: string;
  verb?: string;
  date_added_to_graph?: string;
  [key: string]: any; // Allow any other properties
}

export type NodeType =
  | "Repository"
  | "Language"
  | "Directory"
  | "File"
  | "Import"
  | "Class"
  | "Trait"
  | "Library"
  | "Function"
  | "UnitTest"
  | "IntegrationTest"
  | "E2etest"
  | "Endpoint"
  | "Request"
  | "Datamodel"
  | "Page"
  | "Var"
  | "Message"
  | "Person"
  | "Video"
  | "Hint";

export type EdgeType =
  | "CALLS"
  | "USES"
  | "OPERAND"
  | "CONTAINS"
  | "IMPORTS"
  | "OF"
  | "HANDLER"
  | "RENDERS";

export interface EdgeTypeInterface {
  edge_type: EdgeType;
}
export interface Edge {
  edge: EdgeTypeInterface;
  source: Node;
  target: Node;
}

export interface Neo4jEdge {
  edge_type: string;
  ref_id: string;
  source: string;
  target: string;
  properties: Record<string, any>;
}

export interface WeightedNode {
  name: string;
  relevancy: number;
}

export interface HintExtraction {
  function_names: WeightedNode[];
  file_names: WeightedNode[];
  datamodel_names: WeightedNode[];
  endpoint_names: WeightedNode[];
  page_names: WeightedNode[];
}

export interface GraphResponse {
  nodes: any[];
  edges: any[];
  status: string;
}
export function relevant_node_types(): NodeType[] {
  return [
    "Page",
    "Function",
    "Class",
    "Trait",
    "Datamodel",
    "Request",
    "Endpoint",
    "UnitTest",
    "IntegrationTest",
    "E2etest",
    "Var",
    "Message",
    "Person",
    "Video",
  "Hint",
  ];
}

export function all_node_types(): NodeType[] {
  return [
    "Repository",
    "Language",
    "Directory",
    "File",
    "Import",
    "Class",
    "Trait",
    "Library",
    "Function",
    "UnitTest",
    "IntegrationTest",
    "E2etest",
    "Endpoint",
    "Request",
    "Datamodel",
    "Page",
    "Var",
    "Message",
    "Person",
    "Video",
  "Hint",
  ];
}

export function all_edge_types(): EdgeType[] {
  return [
    "CALLS",
    "USES",
    "OPERAND",
    "CONTAINS",
    "IMPORTS",
    "OF",
    "HANDLER",
    "RENDERS",
  ];
}

export function node_type_descriptions(): { [k in NodeType]: string } {
  return {
    Repository:
      "A code repository that contains source files, directories, and version history.",
    Language: "A programming language used in the repository.",
    Directory:
      "A folder within a repository that organizes files and subdirectories.",
    File: "A file within a repository, containing source code, configuration, or other project-related content.",
    Import:
      "A section at the top of a file that contains all imported modules, libraries, or dependencies used within the file.",
    Class:
      "A class definition in source code, representing an object-oriented structure with attributes and methods.",
    Trait:
      "A trait definition in source code, representing a collection of methods that can be implemented by other classes.",
    Library:
      "A reusable collection of code or modules providing functionality that can be imported and used in other projects.",
    Function:
      "A function or method definition in source code, representing executable logic within a program, including backend logic and frontend components.",
    UnitTest:
      "A unit-level test verifying a single function, component, or module in isolation.",
    IntegrationTest:
      "A test exercising multiple components or an API boundary (e.g. HTTP) without full end-to-end tooling.",
    E2etest:
      "A end-to-end test in source code, representing a complete scenario of user interactions or system operations.",
    Endpoint:
      "A defined entry point for accessing functionality within an application or service, typically through an API.",
    Request: "A request to an specific endpoint",
    Datamodel:
      "A structured representation of data within a system, typically defining entities, relationships, attribute types, and corresponding SQL table definitions.",
    Page: "A webpage or route within an application, representing a specific view or section of the system. It can serve as the starting point for a codemap.",
    Var: "A variable in source code, representing a value that can be used in the code.",
    Message:
      "A message in a conversation between developers, projects managers, or other stakeholders.",
    Person: "A person working on the project.",
    Video: "A recorded video conversation between stakeholders.",
  Hint: "A question and answer pair generated from exploring the codebase to capture contextual understanding.",
  };
}

export function normalizeNodeType(label: string): NodeType | undefined {
  if (label === "Test") return "UnitTest";
  if (label === "E2eTest") return "E2etest";
  return all_node_types().find((t) => t === label) as NodeType | undefined;
}

export function toNum(bi: BoltInt): number {
  if (typeof bi === "object") {
    if (bi.low) {
      return bi.low;
    }
  } else {
    return bi;
  }
  return 0;
}

export interface Service {
  name: string;
  language: string;
  dev: boolean;
  scripts: Record<string, string>;
  env: Record<string, string>;
  pkgFile: string;
}

export interface ServiceParser {
  pkgFileName: string;
  envRegex: RegExp;
  build(pkgFile: Neo4jNode): Service;
}

export interface ContainerConfig {
  name: string;
  config: string; //YAML string for container
}

export enum Language {
  Rust = "rust",
  Go = " go",
  Typescript = "typescript",
  Python = "python",
  Ruby = "ruby",
  Kotlin = "kotlin",
  Swift = "swift",
  Java = "java",
  Cpp = "cpp",
}

export const LANGUAGE_PACKAGE_FILES: Record<Language, string[]> = {
  [Language.Rust]: ["Cargo.toml"],
  [Language.Go]: ["go.mod"],
  [Language.Typescript]: ["package.json"],
  [Language.Python]: ["requirements.txt"],
  [Language.Ruby]: ["Gemfile"],
  [Language.Kotlin]: [".gradle.kts", ".gradle", ".properties"],
  [Language.Swift]: ["Podfile", "Cartfile"],
  [Language.Java]: ["pom.xml"],
  [Language.Cpp]: ["CMakeLists.txt"],
};

export const EXTENSIONS: Record<Language, string[]> = {
  [Language.Rust]: ["rs"],
  [Language.Go]: ["go"],
  [Language.Typescript]: ["ts", "tsx", "js", "jsx"],
  [Language.Python]: ["py", "ipynb"],
  [Language.Ruby]: ["rb"],
  [Language.Kotlin]: ["kt", "kts", "java"],
  [Language.Swift]: ["swift", "plist"],
  [Language.Java]: ["java", "gradle", "gradlew"],
  [Language.Cpp]: ["cpp", "h"],
};

export const LANGUAGE_ENV_REGEX: Record<Language, RegExp> = {
  [Language.Rust]: /std::env::var\("([^"]+)"\)/g,
  [Language.Go]: /os\.(?:Getenv|LookupEnv)\("([^"]+)"\)/g,
  [Language.Typescript]: /process\.env\.(\w+)/g,
  [Language.Python]: /os\.environ\[['"]([^'"]+)['"]\]/g,
  [Language.Ruby]: /ENV\[['"]([^'"]+)['"]\]/g,
  [Language.Kotlin]: /System.getenv\("([^"]+)"\)/g,
  [Language.Swift]: /ProcessInfo\.processInfo\.environment\["([^"]+)"\]/g,
  [Language.Java]: /System\.getProperty\("([^"]+)"\)/g,
  [Language.Cpp]: /getenv\("([^"]+)"\)/g,
};

export const IGNORE_DIRECTORIES = [
  "**/node_modules",
  "**/vendor",
  "**/.git",
  "**/.idea",
  "**/.vscode",
  "**/dist",
  "**/build",
  "**/out",
  "**/target",
  "**/venv",
  "**/logs",
  "**/temp",
  "**/__pycache__",
  "**/.*", //assuming we have no .env in production code
];

export const IGNORE_FILES = [
  "**/*.min.js",
  "**/*.lock",
  "**/*.log",
  "**/*.png",
  "**/*.jpg",
  "**/*.jpeg",
  "**/*.gif",
  "**/*.svg",
  "**/*.ico",
  "**/*.pdf",
  "**/*.zip",
  "**/*.tar",
  "**/*.tar.gz",
  "**/*.rar",
  "**/*.7z",
  "**/*.exe",
  "**/*.dll",
  "**/*.so",
  "**/*.dylib",
  "**/*.bin",
  "**/*.obj",
  "**/*.class",
  "**/*.pyc",
  "**/*.pyo",
  "**/*.db",
  "**/*.sqlite",
  "**/*.mp3",
  "**/*.mp4",
  "**/*.mov",
  "**/*.avi",
  "**/*.mkv",
  "**/*.webm",
  "**/*.ogg",
  "**/*.wav",
  "**/*.flac",
  "**/*.ttf",
  "**/*.woff",
  "**/*.woff2",
  "**/*.eot",
  "**/*.otf",
  "**/*.swp",
  "**/*.swo",
  "**/.DS_Store",
  "**/Thumbs.db",
];
