use std::io::ErrorKind;
use std::path::Path;

use ast::repo::{Repo, Repos};
use ast::Lang;
use console::style;
use lsp::Language;
use shared::{Error, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};

mod args;
mod output;
mod progress;
mod render;

use args::CliArgs;
use output::Output;
use progress::ProgressTracker;
use render::{common_ancestor, first_lines, print_single_file_nodes};

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        if let Error::Io(io_err) = &e {
            if io_err.kind() == ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = CliArgs::parse_and_expand()?;

    let level = if cli.quiet {
        LevelFilter::ERROR
    } else if cli.perf || cli.verbose {
        LevelFilter::INFO
    } else {
        LevelFilter::WARN
    };

    let filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let files = cli.files;
    let allow_unverified_calls = cli.allow;
    let skip_calls = cli.skip_calls;

    let mut out = Output::new();
    let mut files_by_lang: Vec<(Language, Vec<String>)> = Vec::new();
    let mut files_to_print: Vec<String> = Vec::new();

    for file_path in &files {
        if !Path::new(&file_path).exists() {
            return Err(Error::Custom(format!("File does not exist: {}", file_path)));
        }

        let language = Language::from_path(file_path);
        match language {
            Some(lang) => {
                if let Some((_, file_list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                    file_list.push(file_path.clone());
                } else {
                    files_by_lang.push((lang, vec![file_path.clone()]));
                }
                files_to_print.push(file_path.clone());
            }
            None => {
                let contents = std::fs::read_to_string(file_path)?;
                let file_label = style("File:").bold().cyan();
                let msg = format!(
                    "{}  {}\n{}\n",
                    file_label,
                    style(file_path).cyan(),
                    first_lines(&contents, 40, 200)
                );
                out.writeln(msg)?;
            }
        }
    }

    if files_by_lang.is_empty() {
        return Ok(());
    }

    let (progress_tracker, status_tx) = ProgressTracker::new(cli.quiet);
    let progress_handle = tokio::spawn(progress_tracker.run());

    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in files_by_lang.iter() {
        let lang = Lang::from_language(language.clone());

        if let Some(root) = common_ancestor(file_list) {
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            let repo =
                Repo::from_files(&file_refs, root, lang, allow_unverified_calls, skip_calls)?;
            repos_vec.push(repo);
        } else {
            for file_path in file_list {
                let file_lang = Lang::from_language(language.clone());
                let repo = Repo::from_single_file(
                    file_path,
                    file_lang,
                    allow_unverified_calls,
                    skip_calls,
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
        print_single_file_nodes(&mut out, &graph, file_path)?;
    }

    Ok(())
}
