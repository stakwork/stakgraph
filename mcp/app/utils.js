import htm from "https://esm.sh/htm";
import { h } from "https://esm.sh/preact";

export const html = htm.bind(h);

// Constants
export const NODE_TYPE_COLORS = {
  Repository: "#2C5985", // Darker blue
  Language: "#A35D00", // Darker orange
  Directory: "#3A7336", // Darker green
  File: "#ad8cc6", // Darker purple
  Import: "#8B2E2A", // Darker red
  Class: "#4A7D4A", // Darker light green
  Trait: "#3B6EB5", // Darker light blue
  Library: "#A83333", // Darker pink
  Function: "#C67000", // Darker light orange
  Test: "#B7940A", // Darker yellow
  E2etest: "#7C4A85", // Darker lavender
  Endpoint: "#385D8A", // Darker blue gray
  Request: "#6B4A7A", // Darker medium purple
  Datamodel: "#A13939", // Darker salmon
  Page: "#2980B9", // Darker sky blue
};

// Utility functions
export const fileExtension = (filePath) => {
  if (!filePath) return null;
  return filePath.split(".").pop().toLowerCase();
};

export const getLanguageFromFilePath = (filePath) => {
  if (!filePath) return null;
  const extension = filePath.split(".").pop().toLowerCase();
  const langMap = {
    js: "JavaScript",
    jsx: "React",
    ts: "TypeScript",
    tsx: "React TSX",
    py: "Python",
    rb: "Ruby",
    java: "Java",
    go: "Go",
    rs: "Rust",
    c: "C",
    cpp: "C++",
    cs: "C#",
    php: "PHP",
    html: "HTML",
    css: "CSS",
    scss: "SCSS",
    json: "JSON",
    md: "Markdown",
    sql: "SQL",
    sh: "Shell",
    bash: "Bash",
    yaml: "YAML",
    yml: "YAML",
    xml: "XML",
  };

  return langMap[extension] || extension.toUpperCase();
};

export const getHighlightJsClass = (filePath) => {
  if (!filePath) return "";

  const extension = filePath.split(".").pop().toLowerCase();
  const langMap = {
    js: "language-javascript",
    jsx: "language-javascript",
    ts: "language-typescript",
    tsx: "language-typescript",
    py: "language-python",
    rb: "language-ruby",
    java: "language-java",
    go: "language-go",
    rs: "language-rust",
    c: "language-c",
    cpp: "language-cpp",
    cs: "language-csharp",
    php: "language-php",
    html: "language-html",
    css: "language-css",
    scss: "language-scss",
    json: "language-json",
    md: "language-markdown",
    sql: "language-sql",
    sh: "language-bash",
    bash: "language-bash",
    yaml: "language-yaml",
    yml: "language-yaml",
    xml: "language-xml",
  };

  return langMap[extension] || "";
};
