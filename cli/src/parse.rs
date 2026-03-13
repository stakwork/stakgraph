use std::collections::HashSet;
use std::path::Path;

use ast::repo::{Repo, Repos};
use ast::Lang;
use console::style;
use lsp::Language;
use shared::{Error, Result};
use walkdir::WalkDir;

use super::args::CliArgs;
use super::output::Output;
use super::progress::ProgressTracker;
use super::render::print_single_file_nodes;
use super::utils::{common_ancestor, read_text_preview};

fn expand_dirs(inputs: &[String]) -> Result<(Vec<String>, HashSet<String>)> {
    let mut expanded: Vec<String> = Vec::new();
    let mut dir_files: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();

    for input in inputs {
        let path = Path::new(input);
        if path.is_dir() {
            for entry in WalkDir::new(path) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    let p = entry.path().to_string_lossy().to_string();
                    if Language::from_path(&p).is_some() && seen.insert(p.clone()) {
                        expanded.push(p.clone());
                        dir_files.insert(p);
                    }
                }
            }
        } else {
            if seen.insert(input.clone()) {
                expanded.push(input.clone());
            }
        }
    }

    Ok((expanded, dir_files))
}

pub async fn run(cli: &CliArgs, out: &mut Output) -> Result<()> {
    let (files, dir_files) = expand_dirs(&cli.files)?;
    let allow_unverified_calls = cli.allow;
    let skip_calls = cli.skip_calls;
    let no_nested = cli.no_nested;

    let mut files_by_lang: Vec<(Language, Vec<String>)> = Vec::new();
    let mut files_to_print: Vec<String> = Vec::new();

    for file_path in &files {
        if !dir_files.contains(file_path) && !Path::new(file_path).exists() {
            return Err(Error::Custom(format!("File does not exist: {}", file_path)));
        }

        let canonical_path = std::fs::canonicalize(file_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| file_path.clone());
        let language = Language::from_path(file_path);
        match language {
            Some(lang) => {
                if let Some((_, file_list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                    file_list.push(canonical_path.clone());
                } else {
                    files_by_lang.push((lang, vec![canonical_path.clone()]));
                }
                files_to_print.push(canonical_path);
            }
            None => {
                if !dir_files.contains(file_path) {
                    let file_label = style("File:").bold().cyan();
                    let msg = match read_text_preview(file_path) {
                        Some(preview) => format!(
                            "{}  {}\n{}\n",
                            file_label,
                            style(file_path).cyan(),
                            preview
                        ),
                        None => format!(
                            "{}  {}\n[binary or unprintable content skipped]\n",
                            file_label,
                            style(file_path).cyan()
                        ),
                    };
                    out.writeln(msg)?;
                }
            }
        }
    }

    if files_by_lang.is_empty() {
        return Ok(());
    }

    let (progress_tracker, status_tx) = ProgressTracker::new(cli.verbose || cli.perf);
    let progress_handle = tokio::spawn(progress_tracker.run());

    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in files_by_lang.iter() {
        let lang = Lang::from_language(language.clone());

        if let Some(root) = common_ancestor(file_list) {
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            let repo = Repo::from_files(
                &file_refs,
                root,
                lang,
                allow_unverified_calls,
                skip_calls,
                no_nested,
            )?;
            repos_vec.push(repo);
        } else {
            for file_path in file_list {
                let file_lang = Lang::from_language(language.clone());
                let repo = Repo::from_single_file(
                    file_path,
                    file_lang,
                    allow_unverified_calls,
                    skip_calls,
                    no_nested,
                )?;
                repos_vec.push(repo);
            }
        }
    }

    let mut repos = Repos(repos_vec);
    repos.set_status_tx(status_tx).await;

    let graph = repos.build_graphs_array().await?;

    drop(repos);
    let _ = progress_handle.await;

    for file_path in &files_to_print {
        print_single_file_nodes(out, &graph, file_path)?;
    }

    Ok(())
}
