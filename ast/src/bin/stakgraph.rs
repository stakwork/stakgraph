use ast::lang::graphs::NodeType;
use ast::lang::BTreeMapGraph;
use ast::repo::Repo;
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

    // Build name (including verb if present, like "GET /api/users")
    let name = if let Some(verb) = nd.meta.get("verb") {
        format!("{} {}", verb, nd.name)
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
            let body_preview = first_lines(&nd.body, body_lines, 200);
            println!("```\n{}\n```", body_preview);
        }
    }

    // Always print docs if available
    if let Some(docs) = nd.meta.get("docs") {
        println!("Docs: {}", first_lines(docs, 3, 200));
    }
}

fn print_single_file_nodes(graph: &BTreeMapGraph, file_path: &str) -> anyhow::Result<()> {
    let file_path = std::fs::canonicalize(file_path)?
        .to_string_lossy()
        .to_string();
    for node in graph.nodes.values() {
        let node_file = std::fs::canonicalize(&node.node_data.file)
            .unwrap_or_else(|_| std::path::PathBuf::from(&node.node_data.file))
            .to_string_lossy()
            .to_string();
        if node_file == file_path {
            print_node_summary(node);
            println!(); // Add blank line between nodes
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let file_path = args
        .next()
        .ok_or_else(|| Error::Custom("No file path provided".into()))?;
    if !std::path::Path::new(&file_path).exists() {
        return Err(Error::Custom("File does not exist".into()));
    }

    let language = lsp::Language::from_path(&file_path);

    let lang = match language {
        Some(lang) => Lang::from_language(lang),
        None => {
            // If language cannot be determined, output limited file contents
            let contents = std::fs::read_to_string(&file_path)?;
            println!("{}", first_lines(&contents, 40, 200));
            return Ok(());
        }
    };
    let repo = Repo::from_single_file(&file_path, lang)?;
    let graph = repo.build_graph_btree().await?;
    print_single_file_nodes(&graph, &file_path)?;
    Ok(())
}
