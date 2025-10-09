use ast::lang::graphs::NodeType;
use ast::lang::BTreeMapGraph;
use ast::repo::Repo;
use ast::Lang;
use shared::{Error, Result};

/// Limits text to 100 lines, with each line limited to 200 characters
fn limit_output(text: &str) -> String {
    text.lines()
        .take(100)
        .map(|line| {
            if line.len() > 200 {
                format!("{}...", &line[..200])
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn print_node_summary(node: &ast::lang::graphs::Node) {
    let nd = &node.node_data;
    // println!("Node: {:?}", nd.name);
    match &node.node_type {
        NodeType::Function => {
            let lines = if nd.start != nd.end {
                format!("lines {} - {}", nd.start + 1, nd.end + 1)
            } else {
                format!("line {}", nd.start + 1)
            };
            if let Some(interface) = nd.meta.get("interface") {
                println!("Function: {}\n({})", interface, lines);
            } else {
                println!("Function: {} ({})", nd.name, lines);
            }
        }
        NodeType::Endpoint => {
            let verb = nd.meta.get("verb").map(|v| v.as_str()).unwrap_or("");
            println!("Endpoint: {} {}", verb, nd.name);
        }
        NodeType::DataModel | NodeType::Import | NodeType::Request => {
            println!(
                "{}: \n{}",
                node.node_type.to_string(),
                limit_output(&nd.body)
            );
        }
        NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => {
            println!("Test: {}", nd.name);
        }
        NodeType::Directory => {
            // skip directories in summary
        }
        _ => {
            println!("{}: {}", node.node_type.to_string(), nd.name);
        }
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
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // logger();
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
            println!("{}", limit_output(&contents));
            return Ok(());
        }
    };
    let repo = Repo::from_single_file(&file_path, lang)?;
    let graph = repo.build_graph_btree().await?;
    print_single_file_nodes(&graph, &file_path)?;
    Ok(())
}
