use std::path::Path;

use ast::repo::{Repo, Repos};
use ast::Lang;
use lsp::Language;
use shared::{Error, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};

mod args;
mod render;

use args::CliArgs;
use render::{common_ancestor, first_lines, print_single_file_nodes};

#[tokio::main]
async fn main() -> Result<()> {
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
                println!("File: {}\n{}\n", file_path, first_lines(&contents, 40, 200));
            }
        }
    }

    if files_by_lang.is_empty() {
        return Ok(());
    }

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

    let repos = Repos(repos_vec);
    let graph = repos.build_graphs_array().await?;

    for file_path in &files_to_print {
        print_single_file_nodes(&graph, file_path)?;
    }

    Ok(())
}
