use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use console::style;
use lsp::Language;
use serde::Serialize;
use shared::{Error, Result};

use ast::lang::queries::skips::summary::{
    is_junk_file, is_test_file, should_skip_dir, ALWAYS_EXPAND_DIRS, COLLAPSE_DIRS,
    ENTRY_POINT_NAMES, PRIORITY_ROOT_FILES, PRIORITY_SOURCE_DIRS,
};

use super::args::OverviewArgs;
use super::output::{write_json_success, Output, OutputMode};

#[derive(Debug)]
struct DirNode {
    name: String,
    depth: usize,
    dirs: Vec<DirNode>,
    files: Vec<FileEntry>,
    rel_path: String,
}

#[derive(Debug)]
struct FileEntry {
    name: String,
    ext: String,
    is_source: bool,
    rel_path: String,
}

#[derive(Default, Serialize)]
struct OverviewJsonData {
    mode: String,
    root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fingerprint: Option<String>,
    tree: String,
    stats: OverviewJsonStats,
}

#[derive(Default, Serialize)]
struct OverviewJsonStats {
    total_files: usize,
    shown_lines: usize,
    collapsed_dirs: usize,
    hidden_files: usize,
}

enum DirAction {
    Skip,
    Collapse,
    Expand,
}

fn classify_dir(name: &str, depth: usize, total: usize, ext_variety: usize) -> DirAction {
    if depth > 0 && !COLLAPSE_DIRS.contains(&name) && should_skip_dir(name) {
        return DirAction::Skip;
    }
    if COLLAPSE_DIRS.contains(&name) {
        return DirAction::Collapse;
    }
    if ALWAYS_EXPAND_DIRS.contains(&name) && depth <= 2 {
        return DirAction::Expand;
    }
    if total >= 12 && depth >= 2 {
        return DirAction::Collapse;
    }
    if total >= 6 && ext_variety <= 3 && depth >= 2 {
        return DirAction::Collapse;
    }
    if total >= 3 && ext_variety == 1 {
        return DirAction::Collapse;
    }
    DirAction::Expand
}

fn should_include_file(name: &str) -> bool {
    if PRIORITY_ROOT_FILES.contains(&name) || ENTRY_POINT_NAMES.contains(&name) {
        return true;
    }
    if name.starts_with('.') || is_junk_file(name) || is_test_file(name) {
        return false;
    }
    true
}

fn file_ext(name: &str) -> String {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_else(|| "file".to_string())
}

fn build_tree(path: &Path, depth: usize, rel: &str) -> Option<DirNode> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(".")
        .to_string();

    let entries = std::fs::read_dir(path).ok()?;

    let mut dirs = Vec::new();
    let mut files = Vec::new();

    for entry in entries.filter_map(|e| e.ok()) {
        let child_path = entry.path();
        let child_name = child_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let child_rel = if rel.is_empty() {
            child_name.clone()
        } else {
            format!("{}/{}", rel, child_name)
        };

        if child_path.is_dir() {
            if depth > 0
                && should_skip_dir(&child_name)
                && !COLLAPSE_DIRS.contains(&child_name.as_str())
            {
                continue;
            }
            if let Some(sub) = build_tree(&child_path, depth + 1, &child_rel) {
                dirs.push(sub);
            }
        } else if should_include_file(&child_name) {
            let is_source = Language::from_path(child_path.to_str().unwrap_or("")).is_some();
            files.push(FileEntry {
                ext: file_ext(&child_name),
                name: child_name,
                is_source,
                rel_path: child_rel,
            });
        }
    }

    if depth > 0 && dirs.is_empty() && files.is_empty() {
        return None;
    }

    Some(DirNode {
        name,
        depth,
        dirs,
        files,
        rel_path: rel.to_string(),
    })
}

fn count_files(node: &DirNode) -> usize {
    node.files.len() + node.dirs.iter().map(count_files).sum::<usize>()
}

fn count_source_files(node: &DirNode) -> usize {
    node.files.iter().filter(|f| f.is_source).count()
        + node.dirs.iter().map(count_source_files).sum::<usize>()
}

fn collect_ext_counts(node: &DirNode) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for f in &node.files {
        *counts.entry(f.ext.clone()).or_insert(0) += 1;
    }
    for d in &node.dirs {
        for (ext, n) in collect_ext_counts(d) {
            *counts.entry(ext).or_insert(0) += n;
        }
    }
    counts
}

fn score_dir(name: &str, depth: usize, source_count: usize) -> i32 {
    let mut s = (24i32 - depth as i32 * 3).max(0);
    if PRIORITY_SOURCE_DIRS.contains(&name) {
        s += 100;
    }
    if source_count > 0 {
        s += 50;
    }
    if COLLAPSE_DIRS.contains(&name) {
        s += 25;
    }
    s
}

fn score_file(f: &FileEntry, depth: usize) -> i32 {
    let mut s = (24i32 - depth as i32 * 3).max(0);
    if f.is_source {
        s += 50;
    }
    if PRIORITY_ROOT_FILES.contains(&f.name.as_str())
        || ENTRY_POINT_NAMES.contains(&f.name.as_str())
    {
        s += 90;
    }
    if depth == 1 {
        s += 20;
    }
    s
}

fn format_ext_summary(total: usize, ext_counts: &BTreeMap<String, usize>) -> String {
    let mut pairs: Vec<_> = ext_counts.iter().map(|(e, c)| (e.clone(), *c)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    if pairs.len() == 1 {
        let (ext, count) = &pairs[0];
        format!("({} {})", count, ext)
    } else {
        let detail = pairs
            .iter()
            .take(3)
            .map(|(ext, count)| format!("{} {}", count, ext))
            .collect::<Vec<_>>()
            .join(", ");
        format!("({} files: {})", total, detail)
    }
}

fn tree_matches_grep(node: &DirNode, pattern: &str) -> bool {
    let pat = pattern.to_ascii_lowercase();
    for f in &node.files {
        if f.rel_path.to_ascii_lowercase().contains(&pat)
            || f.name.to_ascii_lowercase().contains(&pat)
        {
            return true;
        }
    }
    for d in &node.dirs {
        if d.name.to_ascii_lowercase().contains(&pat) || tree_matches_grep(d, pattern) {
            return true;
        }
    }
    false
}

const JUNK_COLLAPSE: &[&str] = &[
    "node_modules", "vendor", "target", "dist", "build", "out", ".next", "coverage",
    "__pycache__", "obj", ".turbo", ".cache", "storybook-static", "web",
];

fn is_junk_collapse_dir(name: &str) -> bool {
    JUNK_COLLAPSE.contains(&name)
}

fn file_matches_grep(f: &FileEntry, pattern: &str) -> bool {
    let pat = pattern.to_ascii_lowercase();
    f.rel_path.to_ascii_lowercase().contains(&pat) || f.name.to_ascii_lowercase().contains(&pat)
}

fn is_on_zoom_path(dir_rel: &str, zoom_target: &str) -> bool {
    zoom_target.starts_with(dir_rel) || dir_rel.starts_with(zoom_target)
}

fn is_entry_point(name: &str) -> bool {
    ENTRY_POINT_NAMES.contains(&name)
}

fn is_config_file(name: &str) -> bool {
    PRIORITY_ROOT_FILES.contains(&name)
}

fn is_priority_dir(name: &str) -> bool {
    PRIORITY_SOURCE_DIRS.contains(&name)
}

struct RenderOpts<'a> {
    grep: Option<&'a str>,
    zoom: Option<&'a str>,
    marked_files: &'a HashSet<String>,
    use_color: bool,
}

struct RenderState {
    lines: Vec<String>,
    max_lines: usize,
    collapsed_dirs: usize,
    hidden_files: usize,
}

impl RenderState {
    fn new(max_lines: usize) -> Self {
        Self {
            lines: Vec::new(),
            max_lines,
            collapsed_dirs: 0,
            hidden_files: 0,
        }
    }

    fn at_limit(&self) -> bool {
        self.lines.len() >= self.max_lines
    }

    fn push(&mut self, line: String) {
        self.lines.push(line);
    }
}

fn style_dir_name(name: &str, use_color: bool) -> String {
    if !use_color {
        return format!("{}/", name);
    }
    if is_priority_dir(name) {
        format!("{}", style(format!("{}/", name)).cyan().bold())
    } else {
        format!("{}/", name)
    }
}

fn style_file_name(f: &FileEntry, marked: bool, use_color: bool) -> String {
    let marker = if marked { " ●" } else { "" };
    if !use_color {
        return format!("{}{}", f.name, marker);
    }
    let styled_name = if is_entry_point(&f.name) {
        format!("{}", style(&f.name).green().bold())
    } else if is_config_file(&f.name) {
        format!("{}", style(&f.name).yellow())
    } else if f.is_source {
        format!("{}", f.name)
    } else {
        format!("{}", style(&f.name).dim())
    };
    if marked {
        format!("{} {}", styled_name, style("●").green())
    } else {
        styled_name
    }
}

fn style_collapsed_summary(
    name: &str,
    total: usize,
    ext_counts: &BTreeMap<String, usize>,
    use_color: bool,
) -> String {
    let summary = format_ext_summary(total, ext_counts);
    if !use_color {
        return format!("{}/  {}", name, summary);
    }
    format!("{}/  {}", name, style(summary).dim())
}

fn render(
    node: &DirNode,
    prefix: &str,
    is_last: bool,
    is_root: bool,
    state: &mut RenderState,
    opts: &RenderOpts,
) {
    if state.at_limit() {
        state.hidden_files += count_files(node);
        return;
    }

    let in_zoom = opts
        .zoom
        .map(|z| is_on_zoom_path(&node.rel_path, z))
        .unwrap_or(true);
    let is_zoom_target = opts
        .zoom
        .map(|z| node.rel_path == z)
        .unwrap_or(false);

    if is_root {
        state.push(style_dir_name(&node.name, opts.use_color));
    } else {
        let connector = if is_last { "└── " } else { "├── " };
        let total = count_files(node);
        let ext_counts = collect_ext_counts(node);

        if let Some(pat) = opts.grep {
            if !tree_matches_grep(node, pat) {
                return;
            }
        }

        let should_collapse = if is_zoom_target {
            false
        } else if opts.grep.is_some() {
            is_junk_collapse_dir(&node.name)
        } else if opts.zoom.is_some() && !in_zoom {
            total > 0
        } else {
            matches!(
                classify_dir(&node.name, node.depth, total, ext_counts.len()),
                DirAction::Collapse
            )
        };

        match classify_dir(&node.name, node.depth, total, ext_counts.len()) {
            DirAction::Skip => return,
            _ => {}
        }

        if should_collapse {
            if opts.grep.is_some() {
                return;
            }
            state.push(format!(
                "{}{}{}",
                prefix,
                connector,
                style_collapsed_summary(&node.name, total, &ext_counts, opts.use_color)
            ));
            state.collapsed_dirs += 1;
            return;
        }

        state.push(format!(
            "{}{}{}",
            prefix,
            connector,
            style_dir_name(&node.name, opts.use_color)
        ));
    }

    let child_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    let mut scored_dirs: Vec<_> = node
        .dirs
        .iter()
        .map(|d| {
            let sc = count_source_files(d);
            (score_dir(&d.name, d.depth, sc), d)
        })
        .collect();
    scored_dirs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));

    let mut scored_files: Vec<_> = node
        .files
        .iter()
        .filter(|f| {
            if let Some(pat) = opts.grep {
                file_matches_grep(f, pat)
            } else {
                true
            }
        })
        .map(|f| (score_file(f, node.depth + 1), f))
        .collect();
    scored_files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));

    let mut items: Vec<(i32, bool, usize)> = Vec::new();
    for (i, (s, _)) in scored_dirs.iter().enumerate() {
        items.push((*s, true, i));
    }
    for (i, (s, _)) in scored_files.iter().enumerate() {
        items.push((*s, false, i));
    }
    items.sort_by(|a, b| {
        b.0.cmp(&a.0).then_with(|| match (a.1, b.1) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        })
    });

    for (idx_in_list, (_, is_dir, idx)) in items.iter().enumerate() {
        if state.at_limit() {
            break;
        }
        let is_last_child = idx_in_list == items.len() - 1;
        if *is_dir {
            render(
                scored_dirs[*idx].1,
                &child_prefix,
                is_last_child,
                false,
                state,
                opts,
            );
        } else {
            let f = scored_files[*idx].1;
            let connector = if is_last_child { "└── " } else { "├── " };
            let marked = opts.marked_files.contains(&f.rel_path);
            state.push(format!(
                "{}{}{}",
                child_prefix,
                connector,
                style_file_name(f, marked, opts.use_color)
            ));
        }
    }
}

const MANIFEST_FILES: &[(&str, &str)] = &[
    ("Cargo.toml", "Rust"),
    ("package.json", "Node.js"),
    ("go.mod", "Go"),
    ("pyproject.toml", "Python"),
    ("requirements.txt", "Python"),
    ("Gemfile", "Ruby"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java/Kotlin"),
    ("build.gradle.kts", "Kotlin"),
    ("composer.json", "PHP"),
    ("Package.swift", "Swift"),
    ("CMakeLists.txt", "C/C++"),
];

fn detect_fingerprint(root: &Path) -> Option<String> {
    let mut langs = Vec::new();
    let mut dep_names: Vec<String> = Vec::new();

    for (manifest, lang) in MANIFEST_FILES {
        let manifest_path = root.join(manifest);
        if !manifest_path.exists() {
            continue;
        }
        langs.push(*lang);
        let content = std::fs::read_to_string(&manifest_path).unwrap_or_default();

        match *manifest {
            "Cargo.toml" => {
                if content.contains("[workspace]") {
                    if let Some(idx) = langs.iter().position(|l| *l == "Rust") {
                        let members = content
                            .lines()
                            .skip_while(|l| !l.starts_with("members"))
                            .skip(1)
                            .take_while(|l| !l.starts_with(']'))
                            .filter_map(|l| {
                                let trimmed = l.trim().trim_matches('"').trim_matches(',').trim();
                                if trimmed.is_empty() || trimmed == "]" {
                                    None
                                } else {
                                    Some(trimmed.trim_matches('"').to_string())
                                }
                            })
                            .count();
                        if members > 0 {
                            langs[idx] = "Rust";
                            let label = format!("Rust workspace ({} crates)", members);
                            langs[idx] = Box::leak(label.into_boxed_str());
                        }
                    }
                }
                extract_toml_deps(&content, &mut dep_names);
            }
            "package.json" => {
                extract_json_deps(&content, &mut dep_names);
            }
            "go.mod" => {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("require") || trimmed.is_empty() || trimmed == "(" || trimmed == ")" {
                        continue;
                    }
                    if let Some(dep) = trimmed.split_whitespace().next() {
                        if let Some(name) = dep.rsplit('/').next() {
                            dep_names.push(name.to_string());
                        }
                    }
                }
            }
            "pyproject.toml" => {
                extract_toml_deps(&content, &mut dep_names);
            }
            _ => {}
        }
    }

    if langs.is_empty() {
        return None;
    }

    let mut result = langs.join(" · ");

    dep_names.sort();
    dep_names.dedup();
    let top_deps: Vec<_> = dep_names.iter().take(8).cloned().collect();
    if !top_deps.is_empty() {
        result.push_str(&format!("\ndeps: {}", top_deps.join(", ")));
    }

    Some(result)
}

fn extract_toml_deps(content: &str, deps: &mut Vec<String>) {
    let mut in_deps = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[") {
            in_deps = trimmed == "[dependencies]"
                || trimmed == "[dev-dependencies]"
                || trimmed.starts_with("[dependencies.")
                || trimmed == "[tool.poetry.dependencies]";
            continue;
        }
        if in_deps {
            if let Some(name) = trimmed.split('=').next() {
                let name = name.trim();
                if !name.is_empty() && name != "python" {
                    deps.push(name.to_string());
                }
            }
        }
    }
}

fn extract_json_deps(content: &str, deps: &mut Vec<String>) {
    for section in &["dependencies", "devDependencies"] {
        let needle = format!("\"{}\"", section);
        if let Some(start) = content.find(&needle) {
            if let Some(brace) = content[start..].find('{') {
                let from = start + brace + 1;
                if let Some(end) = content[from..].find('}') {
                    let block = &content[from..from + end];
                    for line in block.lines() {
                        let trimmed = line.trim();
                        if let Some(name) = trimmed.strip_prefix('"') {
                            if let Some(end_q) = name.find('"') {
                                deps.push(name[..end_q].to_string());
                            }
                        }
                    }
                }
            }
        }
    }
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn strip_git_prefix(path: &str, git_root: &Path, scan_root: &Path) -> Option<String> {
    let prefix = scan_root.strip_prefix(git_root).ok()?;
    let prefix_str = prefix.to_string_lossy();
    if prefix_str.is_empty() {
        return Some(path.to_string());
    }
    let full_prefix = format!("{}/", prefix_str);
    path.strip_prefix(&*full_prefix).map(|s| s.to_string())
}

fn get_recently_modified_files(scan_root: &Path, n_commits: usize) -> HashSet<String> {
    let git_root = find_git_root(scan_root).unwrap_or_else(|| scan_root.to_path_buf());
    let output = std::process::Command::new("git")
        .args([
            "log",
            "--name-only",
            "--pretty=format:",
            "-n",
            &n_commits.to_string(),
        ])
        .current_dir(&git_root)
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| strip_git_prefix(l, &git_root, scan_root))
            .collect(),
        _ => HashSet::new(),
    }
}

fn get_dirty_files(scan_root: &Path) -> HashSet<String> {
    let git_root = find_git_root(scan_root).unwrap_or_else(|| scan_root.to_path_buf());
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(&git_root)
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| {
                if l.len() > 3 {
                    strip_git_prefix(&l[3..], &git_root, scan_root)
                } else {
                    None
                }
            })
            .collect(),
        _ => HashSet::new(),
    }
}

fn compute_zoom_target(root: &Path, requested: &Path) -> Option<String> {
    let canon_root = root.canonicalize().ok()?;
    let canon_req = requested.canonicalize().ok()?;
    let rel = canon_req.strip_prefix(&canon_root).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    Some(rel.to_string_lossy().to_string())
}

pub fn run(args: &OverviewArgs, out: &mut Output, output_mode: OutputMode) -> Result<()> {
    let raw_root = PathBuf::from(&args.path);
    let root = raw_root
        .canonicalize()
        .map_err(|_| Error::validation(format!("path does not exist: {}", raw_root.display())))?;

    if !root.is_dir() {
        return Err(Error::validation(format!(
            "path is not a directory: {}",
            root.display()
        )));
    }

    let scan_root = if root.join(".git").exists() || has_manifest(&root) {
        root.clone()
    } else {
        find_repo_root(&root).unwrap_or_else(|| root.clone())
    };

    let zoom_target = if scan_root != root {
        compute_zoom_target(&scan_root, &root)
    } else {
        None
    };

    let tree = build_tree(&scan_root, 0, "")
        .ok_or_else(|| Error::validation(format!("empty directory: {}", root.display())))?;

    let mut marked_files = HashSet::new();
    if let Some(n) = args.recent {
        marked_files.extend(get_recently_modified_files(&scan_root, n));
    }
    if args.changed {
        marked_files.extend(get_dirty_files(&scan_root));
    }

    let max_lines = args.max_lines.max(10);
    let use_color = !output_mode.is_json();
    let opts = RenderOpts {
        grep: args.grep.as_deref(),
        zoom: zoom_target.as_deref(),
        marked_files: &marked_files,
        use_color,
    };
    let mut state = RenderState::new(max_lines);
    render(&tree, "", true, true, &mut state, &opts);

    let total_files = count_files(&tree);
    let tree_str = state.lines.join("\n");

    let display_path = {
        let raw = raw_root.to_string_lossy().to_string();
        if raw == "." {
            scan_root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string()
        } else {
            raw
        }
    };

    let fingerprint = if zoom_target.is_none() && args.grep.is_none() {
        detect_fingerprint(&scan_root)
    } else {
        None
    };

    if output_mode.is_json() {
        let data = OverviewJsonData {
            mode: "overview".to_string(),
            root: display_path,
            fingerprint: fingerprint.clone(),
            tree: tree_str,
            stats: OverviewJsonStats {
                total_files,
                shown_lines: state.lines.len(),
                collapsed_dirs: state.collapsed_dirs,
                hidden_files: state.hidden_files,
            },
        };
        write_json_success(out, "overview", data, Vec::new())?;
        return Ok(());
    }

    if let Some(fp) = &fingerprint {
        for line in fp.lines() {
            out.writeln(format!("{}", style(line).cyan().bold()))?;
        }
        out.newline()?;
    }

    out.writeln(&tree_str)?;
    if state.hidden_files > 0 || state.collapsed_dirs > 0 {
        out.newline()?;
        out.writeln(format!(
            "{}",
            style(format!(
                "[{} dirs collapsed, {} files not shown]",
                state.collapsed_dirs, state.hidden_files
            ))
            .dim()
        ))?;
    }
    Ok(())
}

fn has_manifest(path: &Path) -> bool {
    MANIFEST_FILES
        .iter()
        .any(|(name, _)| path.join(name).exists())
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(".git").exists() || has_manifest(&current) {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}
