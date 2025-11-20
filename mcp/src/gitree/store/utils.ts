import { Feature, PRRecord, CommitRecord } from "../types.js";
import { Storage } from "./index.js";

/**
 * Format PR as markdown
 */
export async function formatPRMarkdown(
  pr: PRRecord,
  storage: Storage
): Promise<string> {
  // Get features this PR belongs to
  const features = await storage.getFeaturesForPR(pr.number);
  const featureLinks =
    features.length > 0
      ? `\n---\n\n_Part of features: ${features
          .map((f) => `\`${f.id}\``)
          .join(", ")}_`
      : "";

  const filesList =
    pr.files.length > 0
      ? `\n\n## Files Changed (${pr.files.length})\n\n${formatFilesList(pr.files, pr.newDeclarations)}`
      : "";

  return `# PR #${pr.number}: ${pr.title}

**Merged**: ${pr.mergedAt.toISOString().split("T")[0]}
**URL**: ${pr.url}

## Summary

${pr.summary}${filesList}${featureLinks}
`.trim();
}

/**
 * Format files list with intelligent collapsing and inline declarations
 */
export function formatFilesList(
  files: string[],
  newDeclarations?: PRRecord["newDeclarations"]
): string {
  // Create a map of file -> declarations for easy lookup
  const declMap = new Map<string, string[]>();
  if (newDeclarations) {
    for (const { file, declarations } of newDeclarations) {
      declMap.set(file, declarations);
    }
  }
  // Only collapse if > 20 files total
  if (files.length <= 20) {
    const output: string[] = [];
    for (const file of files) {
      output.push(`- ${file}`);
      const decls = declMap.get(file);
      if (decls) {
        for (const decl of decls) {
          output.push(`  - ${decl}`);
        }
      }
    }
    return output.join("\n");
  }

  // Directories to always collapse
  const autoCollapseDirs = new Set([
    "node_modules",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    "coverage",
  ]);

  // Group files by their full directory path
  const byDirectory: Map<string, string[]> = new Map();
  const rootFiles: string[] = [];

  for (const file of files) {
    const parts = file.split("/");
    if (parts.length === 1) {
      // File in root
      rootFiles.push(file);
    } else {
      // Get the full directory path (everything except the filename)
      const dir = parts.slice(0, -1).join("/");
      if (!byDirectory.has(dir)) {
        byDirectory.set(dir, []);
      }
      byDirectory.get(dir)!.push(file);
    }
  }

  const output: string[] = [];

  // Show root files with their declarations
  for (const file of rootFiles) {
    output.push(`- ${file}`);
    // Add declarations if any
    const decls = declMap.get(file);
    if (decls) {
      for (const decl of decls) {
        output.push(`  - ${decl}`);
      }
    }
  }

  // Process directories
  for (const [dir, dirFiles] of byDirectory.entries()) {
    // Check if any part of the path matches auto-collapse directories
    const pathParts = dir.split("/");
    const shouldAutoCollapse = pathParts.some((part) =>
      autoCollapseDirs.has(part)
    );

    if (shouldAutoCollapse) {
      output.push(`- ${dir}/... (${dirFiles.length} files)`);
      continue;
    }

    // Show first 10, then indicate more
    if (dirFiles.length > 10) {
      // Show first 10 files with declarations
      for (let i = 0; i < 10; i++) {
        output.push(`- ${dirFiles[i]}`);
        const decls = declMap.get(dirFiles[i]);
        if (decls) {
          for (const decl of decls) {
            output.push(`  - ${decl}`);
          }
        }
      }
      // Indicate there are more
      const remaining = dirFiles.length - 10;
      output.push(`- ${dir}/... (${remaining} more files)`);
    } else {
      // Show all files with declarations
      for (const file of dirFiles) {
        output.push(`- ${file}`);
        const decls = declMap.get(file);
        if (decls) {
          for (const decl of decls) {
            output.push(`  - ${decl}`);
          }
        }
      }
    }
  }

  return output.join("\n");
}

/**
 * Parse PR markdown back to PRRecord
 */
export function parsePRMarkdown(number: number, content: string): PRRecord {
  // Simple regex-based parsing for now
  const titleMatch = content.match(/^# PR #\d+: (.+)$/m);
  const mergedMatch = content.match(/\*\*Merged\*\*: (.+)$/m);
  const urlMatch = content.match(/\*\*URL\*\*: (.+)$/m);
  const summaryMatch = content.match(
    /## Summary\n\n([\s\S]+?)(?:\n## Files Changed|\n---|\n*$)/
  );

  // Parse files list and declarations together
  const filesMatch = content.match(
    /## Files Changed \(\d+\)\n\n([\s\S]+?)(?:\n---|\n*$)/
  );
  const files: string[] = [];
  const newDeclarations: PRRecord["newDeclarations"] = [];

  if (filesMatch?.[1]) {
    const lines = filesMatch[1].split("\n");
    let currentFile: string | null = null;
    let currentDecls: string[] = [];

    for (const line of lines) {
      const fileMatch = line.match(/^- (.+)$/);
      const declMatch = line.match(/^  - (.+)$/);

      if (fileMatch) {
        // Save previous file's declarations if any
        if (currentFile && currentDecls.length > 0) {
          newDeclarations.push({ file: currentFile, declarations: currentDecls });
        }
        // Add to files list
        const fileName = fileMatch[1];
        // Skip summary lines like "public/... (88 more files)"
        if (!fileName.includes("...")) {
          files.push(fileName);
        }
        currentFile = fileName;
        currentDecls = [];
      } else if (declMatch && currentFile) {
        currentDecls.push(declMatch[1]);
      }
    }

    // Save last file's declarations
    if (currentFile && currentDecls.length > 0) {
      newDeclarations.push({ file: currentFile, declarations: currentDecls });
    }
  }

  return {
    number,
    title: titleMatch?.[1] || "Unknown",
    mergedAt: mergedMatch?.[1] ? new Date(mergedMatch[1]) : new Date(),
    url: urlMatch?.[1]?.trim() || "",
    summary: summaryMatch?.[1]?.trim() || "",
    files,
    newDeclarations: newDeclarations.length > 0 ? newDeclarations : undefined,
  };
}

/**
 * Format Commit as markdown
 */
export async function formatCommitMarkdown(
  commit: CommitRecord,
  storage: Storage
): Promise<string> {
  // Get features this commit belongs to
  const features = await storage.getFeaturesForCommit(commit.sha);
  const featureLinks =
    features.length > 0
      ? `\n---\n\n_Part of features: ${features
          .map((f) => `\`${f.id}\``)
          .join(", ")}_`
      : "";

  const filesList =
    commit.files.length > 0
      ? `\n\n## Files Changed (${commit.files.length})\n\n${formatFilesList(commit.files, commit.newDeclarations)}`
      : "";

  return `# Commit ${commit.sha.substring(0, 7)}: ${commit.message.split('\n')[0]}

**Author**: ${commit.author}
**Committed**: ${commit.committedAt.toISOString().split("T")[0]}
**URL**: ${commit.url}

## Summary

${commit.summary}${filesList}${featureLinks}
`.trim();
}

/**
 * Parse Commit markdown back to CommitRecord
 */
export function parseCommitMarkdown(sha: string, content: string): CommitRecord {
  // Simple regex-based parsing for now
  const messageMatch = content.match(/^# Commit [a-f0-9]+: (.+)$/m);
  const authorMatch = content.match(/\*\*Author\*\*: (.+)$/m);
  const committedMatch = content.match(/\*\*Committed\*\*: (.+)$/m);
  const urlMatch = content.match(/\*\*URL\*\*: (.+)$/m);
  const summaryMatch = content.match(
    /## Summary\n\n([\s\S]+?)(?:\n## Files Changed|\n---|\n*$)/
  );

  // Parse files list and declarations together
  const filesMatch = content.match(
    /## Files Changed \(\d+\)\n\n([\s\S]+?)(?:\n---|\n*$)/
  );
  const files: string[] = [];
  const newDeclarations: CommitRecord["newDeclarations"] = [];

  if (filesMatch?.[1]) {
    const lines = filesMatch[1].split("\n");
    let currentFile: string | null = null;
    let currentDecls: string[] = [];

    for (const line of lines) {
      const fileMatch = line.match(/^- (.+)$/);
      const declMatch = line.match(/^  - (.+)$/);

      if (fileMatch) {
        // Save previous file's declarations if any
        if (currentFile && currentDecls.length > 0) {
          newDeclarations.push({ file: currentFile, declarations: currentDecls });
        }
        // Add to files list
        const fileName = fileMatch[1];
        // Skip summary lines like "public/... (88 more files)"
        if (!fileName.includes("...")) {
          files.push(fileName);
        }
        currentFile = fileName;
        currentDecls = [];
      } else if (declMatch && currentFile) {
        currentDecls.push(declMatch[1]);
      }
    }

    // Save last file's declarations
    if (currentFile && currentDecls.length > 0) {
      newDeclarations.push({ file: currentFile, declarations: currentDecls });
    }
  }

  return {
    sha,
    message: messageMatch?.[1] || "Unknown",
    author: authorMatch?.[1] || "Unknown",
    committedAt: committedMatch?.[1] ? new Date(committedMatch[1]) : new Date(),
    url: urlMatch?.[1]?.trim() || "",
    summary: summaryMatch?.[1]?.trim() || "",
    files,
    newDeclarations: newDeclarations.length > 0 ? newDeclarations : undefined,
  };
}
