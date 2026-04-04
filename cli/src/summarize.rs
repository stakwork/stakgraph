use std::path::{Path, PathBuf};

use console::style;
use ignore::WalkBuilder;
use shared::{Error, Result};
use tiktoken_rs::cl100k_base;

use ast::lang::graphs::NodeType;
use ast::repo::{Repo, Repos};
use ast::Lang;
use lsp::Language;

use ast::lang::queries::skips::summary::{
    is_junk_file, is_test_file, should_skip_dir, ENTRY_POINT_NAMES,
};

use super::output::Output;
use super::progress::CliSpinner;
use super::render::render_file_nodes_filtered;
use super::utils::rel_path_from_cwd;

const SUMMARY_ALLOWED_TYPES: &[NodeType] = &[
    NodeType::Function,
    NodeType::Class,
    NodeType::DataModel,
    NodeType::Endpoint,
    NodeType::Request,
    NodeType::Trait,
];

fn score_file(path: &Path, root: &Path) -> i32 {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let depth = path
        .strip_prefix(root)
        .map(|p| p.components().count())
        .unwrap_or(99) as i32;
    let entry_bonus = if ENTRY_POINT_NAMES.contains(&name) {
        100
    } else {
        0
    };
    let depth_bonus = (20i32 - depth * 3).max(0);
    entry_bonus + depth_bonus
}

fn count_tokens(bpe: &tiktoken_rs::CoreBPE, s: &str) -> usize {
    bpe.encode_ordinary(console::strip_ansi_codes(s).as_ref())
        .len()
}

fn format_tree(root: &Path, max_depth: usize) -> String {
    let root_name = root.file_name().and_then(|n| n.to_str()).unwrap_or(".");
    let mut lines = vec![format!("{}/", style(root_name).bold().cyan())];
    collect_tree_lines(root, 1, max_depth, &mut lines);
    lines.join("\n")
}

fn collect_tree_lines(dir: &Path, depth: usize, max_depth: usize, lines: &mut Vec<String>) {
    if depth > max_depth {
        return;
    }
    let indent = "  ".repeat(depth);
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if p.is_dir() {
                    !should_skip_dir(name)
                } else {
                    !name.starts_with('.') && !is_junk_file(name) && !is_test_file(name)
                }
            })
            .collect(),
        Err(_) => return,
    };

    entries.sort_by(|a, b| {
        b.is_dir()
            .cmp(&a.is_dir())
            .then_with(|| a.file_name().cmp(&b.file_name()))
    });

    for entry in &entries {
        let name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if entry.is_dir() {
            lines.push(format!("{}{}/", indent, style(name).bold()));
            collect_tree_lines(entry, depth + 1, max_depth, lines);
        } else {
            lines.push(format!("{}{}", indent, name));
        }
    }
}

fn collect_source_files(root: &Path) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true)
        .parents(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                !should_skip_dir(name)
            } else {
                true
            }
        });

    builder
        .build()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            if !e.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                return false;
            }
            let path = e.path();
            if path == root {
                return false;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !is_junk_file(name)
                && !is_test_file(name)
                && !name.starts_with('.')
                && Language::from_path(path.to_str().unwrap_or("")).is_some()
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn collect_md_files(root: &Path) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true)
        .parents(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                !should_skip_dir(name)
            } else {
                true
            }
        });

    let mut files: Vec<(usize, PathBuf)> = builder
        .build()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.path() != root
                && e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("md"))
                    .unwrap_or(false)
        })
        .map(|e| {
            let depth = e.depth();
            (depth, e.path().to_path_buf())
        })
        .collect();
    files.sort_by_key(|(depth, path)| (*depth, path.clone()));
    files.into_iter().map(|(_, p)| p).collect()
}

async fn render_file_summary(file_path: &Path) -> Option<String> {
    let lang = Language::from_path(file_path.to_str()?)?;
    let ast_lang = Lang::from_language(lang);
    let repo = Repo::from_single_file(file_path.to_str()?, ast_lang, false, false)
    .ok()?;

    let graph = Repos(vec![repo]).build_graphs_array().await.ok()?;

    let rendered = render_file_nodes_filtered(&graph, file_path.to_str()?, SUMMARY_ALLOWED_TYPES)
        .ok()?;

    let has_nodes = console::strip_ansi_codes(&rendered)
        .lines()
        .skip(1)
        .any(|l| !l.trim().is_empty());
    if !has_nodes {
        None
    } else {
        Some(rendered)
    }
}

pub async fn run_summarize(
    path: &str,
    max_tokens: usize,
    depth: Option<usize>,
    out: &mut Output,
    show_progress: bool,
) -> Result<()> {
    let spinner = if show_progress {
        Some(CliSpinner::new("Preparing project summary..."))
    } else {
        None
    };
    let bpe = cl100k_base().map_err(|e| Error::Custom(e.to_string()))?;
    let mut tokens_used = 0usize;

    let raw_root = PathBuf::from(path);
    let root = match raw_root.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            if let Some(sp) = &spinner {
                sp.finish_and_clear();
            }
            return Err(Error::validation(format!(
                "path does not exist: {}",
                raw_root.display()
            )));
        }
    };

    if root.is_file() {
        if let Some(sp) = &spinner {
            sp.set_message("Summarizing file structure and key nodes...");
        }
        let header = format!(
            "{} {}",
            style("Summary:").bold(),
            style(rel_path_from_cwd(&root.to_string_lossy())).bold().cyan(),
        );
        out.writeln(&header)?;
        out.newline()?;
        if let Some(rendered) = render_file_summary(&root).await {
            out.writeln(&rendered)?;
        } else {
            out.writeln(style("(no summary — file not parseable or contains no relevant nodes)").dim().to_string())?;
        }
        if let Some(sp) = &spinner {
            sp.finish_with_message("File summary ready");
        }
        return Ok(());
    }

    if !root.is_dir() {
        out.writeln(format!(
            "{}",
            style(format!("Error: {} is not a directory", rel_path_from_cwd(&root.to_string_lossy()))).red()
        ))?;
        return Ok(());
    }

    // ── Header ────────────────────────────────────────────────────────
    let header = format!(
        "{} {}  (budget: {} tokens)",
        style("Summary:").bold(),
        style(rel_path_from_cwd(&root.to_string_lossy())).bold().cyan(),
        style(max_tokens.to_string()).bold().yellow(),
    );
    out.writeln(&header)?;
    out.newline()?;
    tokens_used += count_tokens(&bpe, &header);

    // ── Directory Tree (adaptive depth) ───────────────────────────────
    if let Some(sp) = &spinner {
        sp.set_message("Building directory map...");
    }
    let section = style("Directory Structure").bold().underlined().to_string();
    out.writeln(&section)?;
    tokens_used += count_tokens(&bpe, &section);

    let tree_str = if let Some(d) = depth {
        format_tree(&root, d)
    } else {
        let t1 = format_tree(&root, 1);
        let t1_tok = count_tokens(&bpe, &t1);
        if tokens_used + t1_tok > max_tokens {
            t1
        } else {
            let t2 = format_tree(&root, 2);
            let t2_tok = count_tokens(&bpe, &t2);
            if tokens_used + t2_tok <= max_tokens {
                t2
            } else {
                t1
            }
        }
    };

    tokens_used += count_tokens(&bpe, &tree_str);
    out.writeln(&tree_str)?;
    out.newline()?;

    if tokens_used >= max_tokens {
        if let Some(sp) = &spinner {
            sp.finish_with_message("Summary ready (budget reached)");
        }
        let footer = format!(
            "[{}/{} tokens — budget exhausted at directory tree]",
            tokens_used, max_tokens
        );
        out.writeln(style(footer).dim().to_string())?;
        return Ok(());
    }

    // ── File Summaries ────────────────────────────────────────────────
    if let Some(sp) = &spinner {
        sp.set_message("Summarizing source files...");
    }
    let section = style("File Summaries").bold().underlined().to_string();
    out.writeln(&section)?;
    tokens_used += count_tokens(&bpe, &section);

    let mut source_files = collect_source_files(&root);
    source_files.sort_by(|a, b| score_file(b, &root).cmp(&score_file(a, &root)));

    let mut any_printed = false;
    let mut files_shown = 0usize;
    for (idx, file_path) in source_files.iter().enumerate() {
        if tokens_used >= max_tokens {
            break;
        }
        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        if let Some(sp) = &spinner {
            sp.set_message(format!(
                "Summarizing source files ({}/{}) — {}",
                idx + 1,
                source_files.len(),
                file_name
            ));
        }

        let Some(rendered) = render_file_summary(file_path).await else {
            continue;
        };

        let tok = count_tokens(&bpe, &rendered);
        if tokens_used + tok > max_tokens {
            break;
        }

        out.writeln(&rendered)?;
        tokens_used += tok;
        any_printed = true;
        files_shown += 1;
    }
    let files_skipped = source_files.len().saturating_sub(files_shown);

    if !any_printed {
        out.writeln(
            style("(no summaries — budget too small or no parseable source files found)")
                .dim()
                .to_string(),
        )?;
    }

    // ── Markdown Documentation ─────────────────────────────────────────
    let md_files = collect_md_files(&root);
    if !md_files.is_empty() && tokens_used < max_tokens {
        if let Some(sp) = &spinner {
            sp.set_message("Adding documentation context...");
        }
        out.newline()?;
        let section = style("Documentation").bold().underlined().to_string();
        out.writeln(&section)?;
        tokens_used += count_tokens(&bpe, &section);

        'md: for md_path in &md_files {
            if tokens_used >= max_tokens {
                break;
            }
            let Ok(content) = std::fs::read_to_string(md_path) else {
                continue;
            };
            let name = md_path
                .strip_prefix(&root)
                .unwrap_or(md_path)
                .display()
                .to_string();
            let header = style(format!("Docs: {}", name)).bold().magenta().to_string();
            let header_tok = count_tokens(&bpe, &header);
            if tokens_used + header_tok > max_tokens {
                break;
            }
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() < 10 {
                let total: usize = lines
                    .iter()
                    .map(|l| count_tokens(&bpe, l) + 1)
                    .sum();
                if tokens_used + header_tok + total > max_tokens {
                    continue;
                }
            }

            out.writeln(&header)?;
            tokens_used += header_tok;

            for line in &lines {
                let line_tok = count_tokens(&bpe, line) + 1;
                if tokens_used + line_tok > max_tokens {
                    out.writeln(style("...").dim().to_string())?;
                    break 'md;
                }
                out.writeln(line)?;
                tokens_used += line_tok;
            }
            out.newline()?;
        }
    }

    out.newline()?;
    let footer = if files_skipped > 0 {
        format!(
            "[{}/{} tokens used — {} file{} not shown]",
            tokens_used,
            max_tokens,
            files_skipped,
            if files_skipped == 1 { "" } else { "s" }
        )
    } else {
        format!("[{}/{} tokens used]", tokens_used, max_tokens)
    };
    out.writeln(style(footer).dim().to_string())?;
    if let Some(sp) = &spinner {
        sp.finish_with_message("Project summary ready");
    }

    Ok(())
}
