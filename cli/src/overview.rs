use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
}

#[derive(Debug)]
struct FileEntry {
    name: String,
    ext: String,
    is_source: bool,
}

#[derive(Default, Serialize)]
struct OverviewJsonData {
    mode: String,
    root: String,
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

fn build_tree(path: &Path, depth: usize) -> Option<DirNode> {
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

        if child_path.is_dir() {
            if depth > 0
                && should_skip_dir(&child_name)
                && !COLLAPSE_DIRS.contains(&child_name.as_str())
            {
                continue;
            }
            if let Some(sub) = build_tree(&child_path, depth + 1) {
                dirs.push(sub);
            }
        } else if should_include_file(&child_name) {
            let is_source = Language::from_path(child_path.to_str().unwrap_or("")).is_some();
            files.push(FileEntry {
                ext: file_ext(&child_name),
                name: child_name,
                is_source,
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

fn render(node: &DirNode, indent: usize, state: &mut RenderState) {
    if state.at_limit() {
        state.hidden_files += count_files(node);
        return;
    }

    if indent == 0 {
        state.push(format!("{}/", node.name));
    } else {
        let prefix = "  ".repeat(indent);
        let total = count_files(node);
        let ext_counts = collect_ext_counts(node);

        match classify_dir(&node.name, node.depth, total, ext_counts.len()) {
            DirAction::Skip => return,
            DirAction::Collapse => {
                state.push(format!(
                    "{}{}/  {}",
                    prefix,
                    node.name,
                    format_ext_summary(total, &ext_counts)
                ));
                state.collapsed_dirs += 1;
                return;
            }
            DirAction::Expand => {
                state.push(format!("{}{}/", prefix, node.name));
            }
        }
    }

    let child_indent = indent + 1;

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

    for (_, is_dir, idx) in items {
        if state.at_limit() {
            break;
        }
        if is_dir {
            render(scored_dirs[idx].1, child_indent, state);
        } else {
            let prefix = "  ".repeat(child_indent);
            state.push(format!("{}{}", prefix, scored_files[idx].1.name));
        }
    }
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

    let tree = build_tree(&root, 0)
        .ok_or_else(|| Error::validation(format!("empty directory: {}", root.display())))?;

    let max_lines = args.max_lines.max(10);
    let mut state = RenderState::new(max_lines);
    render(&tree, 0, &mut state);

    let total_files = count_files(&tree);
    let tree_str = state.lines.join("\n");

    let display_path = {
        let raw = raw_root.to_string_lossy().to_string();
        if raw == "." {
            root.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(".")
                .to_string()
        } else {
            raw
        }
    };

    if output_mode.is_json() {
        let data = OverviewJsonData {
            mode: "overview".to_string(),
            root: display_path,
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

    out.writeln(format!("Overview: {}", display_path))?;
    out.newline()?;
    out.writeln(&tree_str)?;
    if state.hidden_files > 0 || state.collapsed_dirs > 0 {
        out.newline()?;
        out.writeln(format!(
            "[{} dirs collapsed, {} files not shown]",
            state.collapsed_dirs, state.hidden_files
        ))?;
    }
    Ok(())
}
