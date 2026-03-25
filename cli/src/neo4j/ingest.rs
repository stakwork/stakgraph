use std::path::Path;

use ast::lang::graphs::graph::Graph;
use ast::repo::{Repo, Repos};
use ast::Lang;
use lsp::Language;
use shared::Result;

use crate::output::Output;
use crate::progress::CliSpinner;
use crate::utils::common_ancestor;

use super::connection::connect_graph_ops;

pub(super) async fn run_ingest(path: &str, out: &mut Output) -> Result<()> {
    let canonical = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string());

    let mut file_list: Vec<String> = Vec::new();
    let root_path = Path::new(&canonical);
    if root_path.is_dir() {
        for entry in walkdir::WalkDir::new(root_path) {
            let entry = entry.map_err(|e| shared::Error::Io(e.into()))?;
            if entry.file_type().is_file() {
                let p = entry.path().to_string_lossy().to_string();
                if Language::from_path(&p).is_some() {
                    file_list.push(p);
                }
            }
        }
    } else if root_path.is_file() {
        file_list.push(canonical.clone());
    }

    if file_list.is_empty() {
        out.writeln("No supported source files found.".to_string())?;
        return Ok(());
    }

    let spinner = CliSpinner::new("Building graph from source files...");

    let mut repos_vec: Vec<Repo> = Vec::new();
    let mut by_lang: std::collections::HashMap<Language, Vec<String>> =
        std::collections::HashMap::new();
    for f in &file_list {
        if let Some(lang) = Language::from_path(f) {
            by_lang.entry(lang).or_default().push(f.clone());
        }
    }
    for (language, files) in by_lang {
        let lang = Lang::from_language(language);
        if let Some(root) = common_ancestor(&files) {
            let file_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
            let repo = Repo::from_files(&file_refs, root, lang, false, false, false)?;
            repos_vec.push(repo);
        }
    }

    let repos = Repos(repos_vec);
    spinner.set_message("Parsing source files...");
    let btree_graph = repos.build_graphs_local().await?;

    let (node_count, edge_count) = {
        let (n, e) = btree_graph.get_graph_size();
        (n, e)
    };

    spinner.set_message("Uploading to Neo4j...");
    let mut ops = connect_graph_ops().await.map_err(|e| {
        spinner.finish_with_message("Failed to connect to Neo4j");
        e
    })?;

    let (final_nodes, final_edges) = ops
        .upload_btreemap_to_neo4j(&btree_graph, None)
        .await
        .map_err(|e| {
            spinner.finish_with_message("Upload failed");
            e
        })?;

    spinner.finish_with_message("Ingest complete");
    out.writeln(format!("Parsed:    {} nodes, {} edges", node_count, edge_count))?;
    out.writeln(format!("Neo4j now: {} nodes, {} edges", final_nodes, final_edges))?;
    Ok(())
}
