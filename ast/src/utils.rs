use std::any::Any;
use std::env;
use std::future::Future;

use crate::lang::graphs::{ArrayGraph, Node};
use crate::lang::{BTreeMapGraph, Graph, NodeRef};
use serde::Serialize;
use shared::Result;
use tokio::runtime::Handle;
use tokio::task;
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::warn;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

pub fn print_json<G: Graph + Serialize + 'static>(graph: &G, name: &str) -> Result<()> {
    let print_root = std::env::var("PRINT_ROOT").unwrap_or_else(|_| "ast/examples".to_string());
    use serde_jsonlines::write_json_lines;
    match std::env::var("OUTPUT_FORMAT")
        .unwrap_or_else(|_| "jsonl".to_string())
        .as_str()
    {
        "jsonl" => {
            if let Some(array_graph) = as_array_graph(graph) {
                let nodepath = format!("{print_root}/{name}-nodes.jsonl");
                write_json_lines(nodepath, &array_graph.nodes)?;
                let edgepath = format!("{print_root}/{name}-edges.jsonl");
                write_json_lines(edgepath, &array_graph.edges)?;
            } else if let Some(btreemap_graph) = as_btreemap_graph(graph) {
                let nodepath = format!("{print_root}/{name}-nodes.jsonl");
                let node_values: Vec<_> = btreemap_graph.nodes.values().collect();
                write_json_lines(nodepath, &node_values)?;
                let edgepath = format!("{print_root}/{name}-edges.jsonl");
                let edge_values = btreemap_graph.get_edges_vec();
                write_json_lines(edgepath, edge_values)?;
            } else {
                //seriolize the whole graph otherwise
                let pretty = serde_json::to_string_pretty(&graph)?;
                let path = format!("{print_root}/{name}.json");
                std::fs::write(path, pretty)?;
            }
        }
        _ => {
            let pretty = serde_json::to_string_pretty(&graph)?;
            let path = format!("{print_root}/{name}.json");
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
    let sanitized_name = sanitize_string(name);

    result.push_str(&sanitize_string(&node_type));
    result.push('-');
    result.push_str(&sanitized_name);
    result.push('-');
    result.push_str(&sanitize_string(file));
    result.push('-');
    result.push_str(&sanitize_string(&start));

    if let Some(v) = meta.get("verb") {
        result.push('-');
        result.push_str(&sanitize_string(v));
    }

    if result.len() > 5000 {
        if sanitized_name.len() > 2000 {
            let truncated_name = &sanitized_name[..2000];
            let mut truncated_result = String::new();
            truncated_result.push_str(&sanitize_string(&node_type));
            truncated_result.push('-');
            truncated_result.push_str(truncated_name);
            truncated_result.push('-');
            truncated_result.push_str(&sanitize_string(file));
            truncated_result.push('-');
            truncated_result.push_str(&sanitize_string(&start));

            if let Some(v) = meta.get("verb") {
                truncated_result.push('-');
                truncated_result.push_str(&sanitize_string(v));
            }

            if truncated_result.len() > 5000 {
                truncated_result.truncate(5000);
            }
            truncated_result
        } else {
            result.truncate(5000);
            result
        }
    } else {
        result
    }
}

pub fn get_use_lsp() -> bool {
    unsafe { env::set_var("LSP_SKIP_POST_CLONE", "true") };

    if let Err(err) = delete_react_testing_node_modules() {
        warn!("failed to clean test node_modules: {err}");
    }
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
    let path = std::path::Path::new("src/testing/typescript/node_modules");
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    let path = std::path::Path::new("src/testing/nextjs/node_modules");
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    let path = std::path::Path::new("/tmp/fayekelmith/demorepo/frontend/node_modules");
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
    let sanitized_name = sanitize_string(name);

    result.push_str(&sanitize_string(&node_type));
    result.push('-');
    result.push_str(&sanitized_name);
    result.push('-');
    result.push_str(&sanitize_string(file));
    result.push('-');
    result.push_str(&sanitize_string(start));

    if let Some(v) = &node_ref.node_data.verb {
        result.push('-');
        result.push_str(&sanitize_string(v));
    }

    if result.len() > 5000 {
        if sanitized_name.len() > 2000 {
            let truncated_name = &sanitized_name[..2000];
            let mut truncated_result = String::new();
            truncated_result.push_str(&sanitize_string(&node_type));
            truncated_result.push('-');
            truncated_result.push_str(truncated_name);
            truncated_result.push('-');
            truncated_result.push_str(&sanitize_string(file));
            truncated_result.push('-');
            truncated_result.push_str(&sanitize_string(start));

            if let Some(v) = &node_ref.node_data.verb {
                truncated_result.push('-');
                truncated_result.push_str(&sanitize_string(v));
            }

            if truncated_result.len() > 5000 {
                truncated_result.truncate(5000);
            }
            truncated_result
        } else {
            result.truncate(5000);
            result
        }
    } else {
        result
    }
}

pub fn sanitize_string(input: &str) -> String {
    input
        .to_lowercase()
        .trim()
        .replace(char::is_whitespace, "")
        .replace(|c: char| !c.is_alphanumeric(), "")
}

// To print Neo4jGraph nodes and edges for testing purposes
pub fn print_json_vec<T: Serialize>(data: &Vec<T>, name: &str) -> Result<()> {
    let file = File::create(format!("ast/examples/{}.jsonl", name))?;
    let mut writer = BufWriter::new(file);
    for item in data {
        serde_json::to_writer(&mut writer, item)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}
pub fn sync_fn<T, F, Fut>(async_fn: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    task::block_in_place(||Handle::current().block_on(async_fn()))
}
