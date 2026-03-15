use std::io::Read;
use std::str::FromStr;

use ast::lang::graphs::NodeType;
use shared::{Error, Result};

pub fn parse_node_types(raw: &[String]) -> Result<Vec<NodeType>> {
    let mut types = Vec::new();
    for s in raw {
        let normalized = match s.to_lowercase().as_str() {
            "repository" => "Repository",
            "package" => "Package",
            "language" => "Language",
            "directory" => "Directory",
            "file" => "File",
            "import" => "Import",
            "library" => "Library",
            "class" => "Class",
            "trait" => "Trait",
            "instance" => "Instance",
            "function" => "Function",
            "endpoint" => "Endpoint",
            "request" => "Request",
            "datamodel" => "Datamodel",
            "feature" => "Feature",
            "page" => "Page",
            "var" => "Var",
            "unittest" => "UnitTest",
            "integrationtest" => "IntegrationTest",
            "e2etest" => "E2etest",
            "mock" => "Mock",
            other => {
                return Err(Error::validation(format!("Unknown node type: '{}'", other)));
            }
        };
        types.push(NodeType::from_str(normalized).map_err(|e| Error::validation(e.to_string()))?);
    }
    Ok(types)
}

pub fn common_ancestor(files: &[String]) -> Option<std::path::PathBuf> {
    if files.is_empty() {
        return None;
    }

    let mut ancestors: Vec<Vec<std::path::PathBuf>> = Vec::new();
    for file in files {
        let abs_path = std::fs::canonicalize(file).ok()?;
        let mut path_ancestors = Vec::new();
        let mut current = abs_path.as_path();
        while let Some(parent) = current.parent() {
            path_ancestors.push(parent.to_path_buf());
            current = parent;
        }
        path_ancestors.reverse();
        ancestors.push(path_ancestors);
    }

    if ancestors.is_empty() {
        return None;
    }

    let mut common = std::path::PathBuf::new();
    let min_len = ancestors.iter().map(|a| a.len()).min()?;

    for i in 0..min_len {
        let dir = &ancestors[0][i];
        if ancestors.iter().all(|a| a.get(i) == Some(dir)) {
            common = dir.clone();
        } else {
            break;
        }
    }

    if common.as_os_str().is_empty() {
        None
    } else {
        Some(common)
    }
}

const PREVIEW_BYTES: usize = 40_960;
const PREVIEW_CHARS: usize = 10_000;

pub fn read_text_preview(file_path: &str) -> Option<String> {
    if let Some(ext) = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
    {
        if lsp::language::common_binary_exts().contains(&ext) {
            return None;
        }
    }

    let mut buf = vec![0u8; PREVIEW_BYTES];
    let n = {
        let mut f = std::fs::File::open(file_path).ok()?;
        f.read(&mut buf).ok()?
    };
    let bytes = &buf[..n];

    if bytes.contains(&0u8) {
        return None;
    }

    let text = String::from_utf8(bytes.to_vec()).ok()?;
    if text.chars().count() <= PREVIEW_CHARS {
        Some(text)
    } else {
        let mut end = 0;
        let mut count = 0;
        for c in text.chars() {
            if count >= PREVIEW_CHARS {
                break;
            }
            end += c.len_utf8();
            count += 1;
        }
        Some(text[..end].to_string())
    }
}

pub fn first_lines(text: &str, n: usize, max_line_len: usize) -> String {
    text.lines()
        .take(n)
        .map(|line| line.chars().take(max_line_len).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}
