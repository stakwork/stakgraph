use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use ast::lang::graphs::{ArrayGraph, NodeType};
use ast::lang::Lang;
use ast::repo::{Repo, Repos};
use lsp::Language;
use shared::{Error, Result};
use walkdir::WalkDir;

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

pub fn expand_dirs_for_parse(inputs: &[String]) -> Vec<String> {
    let mut expanded: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for input in inputs {
        let path = Path::new(input);
        if path.is_dir() {
            for entry in WalkDir::new(path).into_iter().flatten() {
                if entry.file_type().is_file() {
                    let raw = entry.path().to_string_lossy().to_string();
                    let p = std::fs::canonicalize(&raw)
                        .map(|c| c.to_string_lossy().to_string())
                        .unwrap_or(raw);
                    if Language::from_path(&p).is_some() && seen.insert(p.clone()) {
                        expanded.push(p);
                    }
                }
            }
        } else {
            let p = std::fs::canonicalize(input)
                .map(|c| c.to_string_lossy().to_string())
                .unwrap_or_else(|_| input.clone());
            if seen.insert(p.clone()) {
                expanded.push(p);
            }
        }
    }

    expanded
}

pub async fn build_graph_for_files(files: &[String]) -> Result<ArrayGraph> {
    build_graph_for_files_with_options(files, true).await
}

pub async fn build_graph_for_files_with_options(
    files: &[String],
    allow_unverified_calls: bool,
) -> Result<ArrayGraph> {
    if files.is_empty() {
        return Ok(ArrayGraph::default());
    }

    let mut files_by_lang: Vec<(Language, Vec<String>)> = Vec::new();
    for file_path in files {
        if let Some(lang) = Language::from_path(file_path) {
            if let Some((_, list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                list.push(file_path.clone());
            } else {
                files_by_lang.push((lang, vec![file_path.clone()]));
            }
        }
    }

    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in &files_by_lang {
        let lang = Lang::from_language(language.clone());
        if let Some(root) = common_ancestor(file_list) {
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            repos_vec.push(Repo::from_files(
                &file_refs,
                root,
                lang,
                allow_unverified_calls,
                false,
            )?);
        } else {
            for file_path in file_list {
                let file_lang = Lang::from_language(language.clone());
                repos_vec.push(Repo::from_single_file(
                    file_path,
                    file_lang,
                    allow_unverified_calls,
                    false,
                )?);
            }
        }
    }

    if repos_vec.is_empty() {
        return Ok(ArrayGraph::default());
    }

    let repos = Repos(repos_vec);
    repos.build_graphs_array().await
}

pub fn rel_path_from_cwd(path: &str) -> String {
    let base = std::env::current_dir().unwrap_or_default();
    Path::new(path)
        .strip_prefix(&base)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string())
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
