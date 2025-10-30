use ast::lang::graphs::NodeType;
use ast::lang::ArrayGraph;
use ast::lang::graphs::EdgeType;
use ast::repo::{Repo, Repos};
use ast::Lang;
use shared::{Error, Result};

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

    // Print nodes in order
    for node in nodes {
        print_node_summary(node);
        
        if matches!(node.node_type, NodeType::Function) {
            let source_key = ast::utils::create_node_key(node).to_lowercase();
            
            for edge in &graph.edges {
                let edge_source_key = ast::utils::create_node_key_from_ref(&edge.source).to_lowercase();
                
                if edge_source_key == source_key {
                    if matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
                        let target_name = &edge.target.node_data.name;
                        let target_line = edge.target.node_data.start;
                        
                        println!("  • {:?} → {} (L{})", edge.edge, target_name, target_line + 1);
                        
                    }
                }
            }
        }
        
        println!(); // Add blank line between nodes
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<String> = Vec::new();
    let mut allow_unverified_calls = false;

    for a in raw_args {
        if a == "--allow" {
            allow_unverified_calls = true;
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

    let mut repos_vec: Vec<Repo> = Vec::new();
    let mut files_to_print: Vec<String> = Vec::new();

    for file_path in &files {
        if !std::path::Path::new(&file_path).exists() {
            return Err(Error::Custom(format!("File does not exist: {}", file_path)));
        }

        let language = lsp::Language::from_path(&file_path);
        match language {
            Some(lang) => {
                let lang = Lang::from_language(lang);
                let repo = Repo::from_single_file(&file_path, lang, allow_unverified_calls)?;
                repos_vec.push(repo);
                files_to_print.push(file_path.clone());
            }
            None => {
                let contents = std::fs::read_to_string(&file_path)?;
                println!("File: {}\n{}\n", file_path, first_lines(&contents, 40, 200));
            }
        }
    }
    if repos_vec.is_empty() {
        return Ok(());
    }

    let repos = Repos(repos_vec);
    let graph = repos.build_graphs_array().await?;

    for file_path in &files_to_print {
        print_single_file_nodes(&graph, file_path)?;
    }

    Ok(())
}
