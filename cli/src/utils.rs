use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use ast::lang::graphs::{ArrayGraph, NodeType};
use ast::lang::Lang;
use ast::repo::{Repo, Repos};
use globset::{Glob, GlobSet, GlobSetBuilder};
use lsp::Language;
use shared::{Error, Result};
use walkdir::WalkDir;

fn has_glob_chars(pattern: &str) -> bool {
    pattern
        .chars()
        .any(|ch| matches!(ch, '*' | '?' | '[' | ']' | '{' | '}'))
}

fn expand_glob_pattern(raw_pattern: &str) -> Vec<String> {
    let mut pattern = raw_pattern.trim().replace('\\', "/");
    while pattern.starts_with("./") {
        pattern = pattern.trim_start_matches("./").to_string();
    }
    pattern = pattern.trim_start_matches('/').to_string();

    if pattern.is_empty() {
        return Vec::new();
    }

    if pattern.ends_with('/') {
        pattern.push_str("**");
    }

    let has_glob = has_glob_chars(&pattern);

    if !pattern.contains('/') {
        if has_glob {
            return vec![format!("**/{}", pattern)];
        }
        return vec![format!("**/{}", pattern), format!("**/{}/**", pattern)];
    }

    if !has_glob {
        let base = pattern.trim_end_matches('/');
        return vec![base.to_string(), format!("{}/**", base)];
    }

    vec![pattern]
}

fn compile_globset(patterns: &[String], kind: &str) -> Result<Option<GlobSet>> {
    let mut builder = GlobSetBuilder::new();
    let mut count = 0usize;

    for raw in patterns {
        for expanded in expand_glob_pattern(raw) {
            let glob = Glob::new(&expanded).map_err(|e| {
                Error::validation(format!("invalid {} glob pattern '{}': {}", kind, raw, e))
            })?;
            builder.add(glob);
            count += 1;
        }
    }

    if count == 0 {
        return Ok(None);
    }

    let built = builder
        .build()
        .map_err(|e| Error::validation(format!("invalid {} glob patterns: {}", kind, e)))?;
    Ok(Some(built))
}

fn normalize_path_for_match(path: &str, cwd: &Path) -> String {
    let input_path = Path::new(path);
    let absolute = std::fs::canonicalize(input_path).unwrap_or_else(|_| {
        if input_path.is_absolute() {
            input_path.to_path_buf()
        } else {
            cwd.join(input_path)
        }
    });

    let rel = absolute.strip_prefix(cwd).unwrap_or(&absolute);
    rel.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_start_matches('/')
        .to_string()
}

fn matches_any(set: &GlobSet, normalized_path: &str) -> bool {
    if set.is_match(normalized_path) {
        return true;
    }
    let basename = Path::new(normalized_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    !basename.is_empty() && set.is_match(&basename)
}

pub fn apply_glob_filters(
    files: Vec<String>,
    include_patterns: &[String],
    exclude_patterns: &[String],
) -> Result<Vec<String>> {
    let cwd = std::env::current_dir()
        .map_err(|e| Error::internal(format!("Failed to get current directory: {}", e)))?;
    apply_glob_filters_with_base(files, include_patterns, exclude_patterns, &cwd)
}

fn apply_glob_filters_with_base(
    files: Vec<String>,
    include_patterns: &[String],
    exclude_patterns: &[String],
    base_dir: &Path,
) -> Result<Vec<String>> {
    if files.is_empty() {
        return Ok(files);
    }

    let include_set = compile_globset(include_patterns, "include")?;
    let exclude_set = compile_globset(exclude_patterns, "exclude")?;

    Ok(files
        .into_iter()
        .filter(|file| {
            let normalized = normalize_path_for_match(file, base_dir);
            let included = include_set
                .as_ref()
                .map(|set| matches_any(set, &normalized))
                .unwrap_or(true);
            if !included {
                return false;
            }
            let excluded = exclude_set
                .as_ref()
                .map(|set| matches_any(set, &normalized))
                .unwrap_or(false);
            !excluded
        })
        .collect())
}

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
            "concept" => "Concept",
            "feature" => "Concept", // legacy alias
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

pub fn expand_dirs_for_parse_with_globs(
    inputs: &[String],
    include_patterns: &[String],
    exclude_patterns: &[String],
) -> Result<Vec<String>> {
    let expanded = expand_dirs_for_parse(inputs);
    apply_glob_filters(expanded, include_patterns, exclude_patterns)
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

/// Returns true when `full` equals `suffix` or `full` ends with `/<suffix>`.
/// Avoids false positives like "src/lib.rs" matching "othersrc/lib.rs".
pub fn path_suffix_matches(full: &str, suffix: &str) -> bool {
    full == suffix
        || full
            .strip_suffix(suffix)
            .is_some_and(|rest| rest.ends_with('/'))
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

#[cfg(test)]
mod tests {
    use super::{apply_glob_filters, apply_glob_filters_with_base};
    use std::fs;

    fn mk_file(path: &std::path::Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("failed to create dir");
        }
        fs::write(path, body).expect("failed to write file");
    }

    #[test]
    fn include_glob_matches_nested_paths() {
        let td = tempfile::tempdir().expect("tempdir");
        let root = td.path();
        mk_file(&root.join("src/api/users.ts"), "export const users = 1;");
        mk_file(&root.join("src/lib/math.ts"), "export const sum = 1;");

        let files = vec![
            root.join("src/api/users.ts").to_string_lossy().to_string(),
            root.join("src/lib/math.ts").to_string_lossy().to_string(),
        ];
        let filtered = apply_glob_filters_with_base(files, &["**/api/**".to_string()], &[], root)
            .expect("filter");

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].ends_with("users.ts"));
    }

    #[test]
    fn exclude_wins_over_include() {
        let td = tempfile::tempdir().expect("tempdir");
        let root = td.path();
        mk_file(&root.join("src/api/users.ts"), "u");
        mk_file(&root.join("src/api/comments.ts"), "c");

        let files = vec![
            root.join("src/api/users.ts").to_string_lossy().to_string(),
            root.join("src/api/comments.ts").to_string_lossy().to_string(),
        ];
        let filtered = apply_glob_filters_with_base(
            files,
            &["**/api/**".to_string()],
            &["**/users/**".to_string(), "**/users.ts".to_string()],
            root,
        )
        .expect("filter");

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].ends_with("comments.ts"));
    }

    #[test]
    fn plain_segment_pattern_matches_like_vscode() {
        let td = tempfile::tempdir().expect("tempdir");
        let root = td.path();
        mk_file(&root.join("src/tests/sample.ts"), "s");
        mk_file(&root.join("src/lib/sample.ts"), "s");

        let files = vec![
            root.join("src/tests/sample.ts").to_string_lossy().to_string(),
            root.join("src/lib/sample.ts").to_string_lossy().to_string(),
        ];
        let filtered =
            apply_glob_filters_with_base(files, &[], &["tests".to_string()], root).expect("filter");

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].ends_with("src/lib/sample.ts"));
    }

    #[test]
    fn invalid_pattern_returns_validation_error() {
        let res = apply_glob_filters(
            vec!["src/lib/file.ts".to_string()],
            &["[".to_string()],
            &[],
        );
        assert!(res.is_err());
    }
}
