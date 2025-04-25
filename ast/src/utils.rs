use std::any::Any;
use std::env;

use crate::lang::graphs::{ArrayGraph, Node};
use crate::lang::{BTreeMapGraph, Graph, NodeRef};
use anyhow::Result;
use serde::Serialize;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

pub fn print_json<G: Graph + Serialize + 'static>(graph: &G, name: &str) -> Result<()> {
    use serde_jsonlines::write_json_lines;
    match std::env::var("OUTPUT_FORMAT")
        .unwrap_or_else(|_| "json".to_string())
        .as_str()
    {
        "jsonl" => {
            if let Some(array_graph) = as_array_graph(graph) {
                let nodepath = format!("ast/examples/{}-nodes.jsonl", name);
                write_json_lines(nodepath, &array_graph.nodes)?;
                let edgepath = format!("ast/examples/{}-edges.jsonl", name);
                write_json_lines(edgepath, &array_graph.edges)?;
            } else if let Some(btreemap_graph) = as_btreemap_graph(graph) {
                let nodepath = format!("ast/examples/{}-nodes.jsonl", name);
                write_json_lines(nodepath, &btreemap_graph.nodes)?;
                let edgepath = format!("ast/examples/{}-edges.jsonl", name);
                write_json_lines(edgepath, &btreemap_graph.edges)?;
            } else {
                //seriolize the whole graph otherwise
                let pretty = serde_json::to_string_pretty(&graph)?;
                let path = format!("ast/examples/{}.json", name);
                std::fs::write(path, pretty)?;
            }
        }
        _ => {
            let pretty = serde_json::to_string_pretty(&graph)?;
            let path = format!("ast/examples/{}.json", name);
            std::fs::write(path, pretty)?;
        }
    }
    Ok(())
}

fn as_array_graph<G: Graph + Serialize + 'static>(graph: &G) -> Option<&ArrayGraph> {
    (graph as &dyn Any).downcast_ref::<ArrayGraph>()
}

fn as_btreemap_graph<G: Graph + Serialize + 'static>(graph: &G) -> Option<&BTreeMapGraph> {
    (graph as &dyn Any).downcast_ref::<BTreeMapGraph>()
}

pub fn logger() {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();
}

pub fn create_node_key(node: &Node) -> String {
    let node_type = node.node_type.to_string();
    let node_data = &node.node_data;
    let name = &node_data.name;
    let file = &node_data.file;
    let start = node_data.start.to_string();
    let meta = &node_data.meta;

    let mut result = String::new();

    result.push_str(&sanitize_string(&node_type));
    result.push('-');
    result.push_str(&sanitize_string(name));
    result.push('-');
    result.push_str(&sanitize_string(file));
    result.push('-');
    result.push_str(&sanitize_string(&start));

    if let Some(v) = meta.get("verb") {
        result.push('-');
        result.push_str(&sanitize_string(v));
    }
    result
}

pub fn get_use_lsp() -> bool {
    println!("===-==> Getting use LSP");
    env::set_var("LSP_SKIP_POST_CLONE", "true");
    delete_react_testing_node_modules().ok();
    let lsp = env::var("USE_LSP").unwrap_or_else(|_| "false".to_string());
    if lsp == "true" || lsp == "1" {
        return true;
    }
    false
}

fn delete_react_testing_node_modules() -> std::io::Result<()> {
    let path = std::path::Path::new("src/testing/react/node_modules");
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}
pub fn create_node_key_from_ref(node_ref: &NodeRef) -> String {
    let node_type = node_ref.node_type.to_string().to_lowercase();
    let name = &node_ref.node_data.name;
    let file = &node_ref.node_data.file;
    let start = &node_ref.node_data.start.to_string();

    let mut result = String::new();

    result.push_str(&sanitize_string(&node_type));
    result.push('-');
    result.push_str(&sanitize_string(name));
    result.push('-');
    result.push_str(&sanitize_string(file));
    result.push('-');
    result.push_str(&sanitize_string(&start));

    if let Some(v) = &node_ref.node_data.verb {
        result.push('-');
        result.push_str(&sanitize_string(v));
    }

    result
}

pub fn sanitize_string(input: &str) -> String {
    input
        .to_lowercase()
        .trim()
        .replace(char::is_whitespace, "")
        .replace(|c: char| !c.is_alphanumeric(), "")
}
