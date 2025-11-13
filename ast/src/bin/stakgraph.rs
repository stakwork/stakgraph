use ast::lang::graphs::EdgeType;
use ast::lang::graphs::NodeType;
use ast::lang::ArrayGraph;
use ast::repo::{Repo, Repos};
use ast::Lang;
use shared::{Error, Result};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

/// Compute the common ancestor directory for a list of file paths
fn common_ancestor(files: &[String]) -> Option<std::path::PathBuf> {
    if files.is_empty() {
        return None;
    }

    // Get the absolute paths and their ancestors
    let mut ancestors: Vec<Vec<std::path::PathBuf>> = Vec::new();
    for file in files {
        let abs_path = std::fs::canonicalize(file).ok()?;
        let mut path_ancestors = Vec::new();
        let mut current = abs_path.as_path();
        while let Some(parent) = current.parent() {
            path_ancestors.push(parent.to_path_buf());
            current = parent;
        }
        path_ancestors.reverse(); // root first
        ancestors.push(path_ancestors);
    }

    if ancestors.is_empty() {
        return None;
    }

    // Find the longest common prefix
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

/// Limits text to n lines, with each line limited to max_line_len characters
fn first_lines(text: &str, n: usize, max_line_len: usize) -> String {
    text.lines()
        .take(n)
        .map(|line| {
            if line.len() > max_line_len {
                &line[..max_line_len]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format line numbers as "start-end" or just "start" if single line
fn format_lines(start: usize, end: usize) -> String {
    if start != end {
        format!("{}-{}", start + 1, end + 1)
    } else {
        format!("{}", start + 1)
    }
}

fn print_node_summary(node: &ast::lang::graphs::Node) {
    let nd = &node.node_data;

    let name = if let Some(verb) = nd.meta.get("verb") {
        // Build name (including verb if present, like "GET /api/users")
        format!("{} {}", verb, nd.name)
    } else if matches!(node.node_type, NodeType::Import) {
        // skip Import node name
        "".to_string()
    } else {
        nd.name.clone()
    };

    let lines = format_lines(nd.start, nd.end);

    // Always print: NodeType: name (lines)
    println!("{}: {} ({})", node.node_type.to_string(), name, lines);

    // Show interface if available, otherwise show body
    if let Some(interface) = nd.meta.get("interface") {
        println!("```\n{}\n```", interface);
    } else {
        // Determine how many lines of body to show based on node type
        let body_lines = match node.node_type {
            NodeType::Function | NodeType::Endpoint | NodeType::Var => 20,
            NodeType::DataModel | NodeType::Import | NodeType::Request => 100,
            NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => 0,
            _ => 0,
        };

        if body_lines > 0 && !nd.body.is_empty() {
            let body = if matches!(node.node_type, NodeType::Import) {
                // Remove empty lines from Import nodes
                nd.body
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                nd.body.clone()
            };
            let body_preview = first_lines(&body, body_lines, 200);
            println!("```\n{}\n```", body_preview);
        }
    }

    if let Some(docs) = nd.meta.get("docs") {
        println!("Docs: {}", first_lines(docs, 3, 200));
    }
}

fn print_single_file_nodes(graph: &ArrayGraph, file_path: &str) -> anyhow::Result<()> {
    let file_path = std::fs::canonicalize(file_path)?
        .to_string_lossy()
        .to_string();

    println!("File: {}", file_path);

    // Collect matching nodes
    let mut nodes: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                return false;
            }
            let node_file = std::fs::canonicalize(&node.node_data.file)
                .unwrap_or_else(|_| std::path::PathBuf::from(&node.node_data.file))
                .to_string_lossy()
                .to_string();
            node_file == file_path
        })
        .collect();

    // Sort by start line
    nodes.sort_by_key(|node| node.node_data.start);

    // Build an index of edges by source key for faster lookup
    let mut edges_by_source: std::collections::HashMap<String, Vec<&ast::lang::graphs::Edge>> =
        std::collections::HashMap::new();
    for edge in &graph.edges {
        if matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
            let source_key = ast::utils::create_node_key_from_ref(&edge.source).to_lowercase();
            edges_by_source
                .entry(source_key)
                .or_insert_with(Vec::new)
                .push(edge);
        }
    }

    // Print nodes in order
    for node in nodes {
        print_node_summary(node);

        if matches!(node.node_type, NodeType::Function) {
            let source_key = ast::utils::create_node_key(node).to_lowercase();
            let source_file = &node.node_data.file;

            // Look up edges for this function using the index
            if let Some(edges) = edges_by_source.get(&source_key) {
                for edge in edges {
                    let target_name = &edge.target.node_data.name;
                    let target_line = edge.target.node_data.start;
                    let target_file = &edge.target.node_data.file;

                    // Check if target is in a different file
                    let file_info = if source_file != target_file {
                        // Extract just the filename
                        let filename = std::path::Path::new(target_file)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(target_file);
                        format!(" [{}]", filename)
                    } else {
                        String::new()
                    };

                    println!(
                        "  • {:?} → {} (L{}){}",
                        edge.edge,
                        target_name,
                        target_line + 1,
                        file_info
                    );
                }
            }
        }

        println!(); // Add blank line between nodes
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::ERROR.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<String> = Vec::new();
    let mut allow_unverified_calls = false;
    let mut skip_calls = false;

    for a in raw_args {
        if a == "--allow" {
            allow_unverified_calls = true;
            continue;
        }
        if a == "--skip-calls" {
            skip_calls = true;
            continue;
        }
        for part in a.split(',') {
            let p = part.trim();
            if !p.is_empty() {
                files.push(p.to_string());
            }
        }
    }

    if files.is_empty() {
        return Err(Error::Custom("No file path provided".into()));
    }

    // Group files by language
    let mut files_by_lang: Vec<(lsp::Language, Vec<String>)> = Vec::new();
    let mut files_to_print: Vec<String> = Vec::new();

    for file_path in &files {
        if !std::path::Path::new(&file_path).exists() {
            return Err(Error::Custom(format!("File does not exist: {}", file_path)));
        }

        let language = lsp::Language::from_path(&file_path);
        match language {
            Some(lang) => {
                // Find or create language bucket
                if let Some((_, file_list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                    file_list.push(file_path.clone());
                } else {
                    files_by_lang.push((lang, vec![file_path.clone()]));
                }
                files_to_print.push(file_path.clone());
            }
            None => {
                let contents = std::fs::read_to_string(&file_path)?;
                println!("File: {}\n{}\n", file_path, first_lines(&contents, 40, 200));
            }
        }
    }

    if files_by_lang.is_empty() {
        return Ok(());
    }

    // Create one Repo per language
    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in files_by_lang.iter() {
        let lang = Lang::from_language(language.clone());

        // Try to find a common ancestor for all files
        if let Some(root) = common_ancestor(file_list) {
            // All files share a common ancestor, create a single Repo
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            let repo =
                Repo::from_files(&file_refs, root, lang, allow_unverified_calls, skip_calls)?;
            repos_vec.push(repo);
        } else {
            // No common ancestor, create individual repos per file
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
