use std::any::Any;
use std::env;

use crate::lang::graphs::{ArrayGraph, Node};
use crate::lang::{BTreeMapGraph, Graph, NodeRef};
use serde::Serialize;
use shared::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;
use std::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub enum CallFinderStrategy {
    OnlyOne,
    SameFile,
    Import,
    SameDir,
    Operand,
    NestedVar,
    NotFound,
}

#[derive(Default, Debug)]
pub struct CallFinderStats {
    pub total_lookups: usize,
    pub total_time_ms: u128,
    pub only_one_hits: usize,
    pub same_file_hits: usize,
    pub import_hits: usize,
    pub same_dir_hits: usize,
    pub operand_hits: usize,
    pub nested_var_hits: usize,
    pub not_found: usize,
}

thread_local! {
    static CALL_FINDER_STATS: RefCell<CallFinderStats> = RefCell::new(CallFinderStats::default());
}



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
    result.push_str(&sanitize_string(&start));

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
            truncated_result.push_str(&sanitize_string(&start));
            
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
    Fut: std::future::Future<Output = T>,
{
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async_fn()))
}

pub fn record_call_finder_lookup(elapsed_ms: u128, strategy: CallFinderStrategy) {
    CALL_FINDER_STATS.with(|stats| {
        let mut s = stats.borrow_mut();
        s.total_lookups += 1;
        s.total_time_ms += elapsed_ms;
        match strategy {
            CallFinderStrategy::OnlyOne => s.only_one_hits += 1,
            CallFinderStrategy::SameFile => s.same_file_hits += 1,
            CallFinderStrategy::Import => s.import_hits += 1,
            CallFinderStrategy::SameDir => s.same_dir_hits += 1,
            CallFinderStrategy::Operand => s.operand_hits += 1,
            CallFinderStrategy::NestedVar => s.nested_var_hits += 1,
            CallFinderStrategy::NotFound => s.not_found += 1,
        }
    });
}



pub fn log_and_reset_call_finder_stats() {
    CALL_FINDER_STATS.with(|stats| {
        let s = stats.borrow();
        if s.total_lookups > 0 {
            tracing::info!(
                "[perf][call_finder] lookups={} time_ms={} only_one={} same_file={} import={} same_dir={} operand={} nested={} not_found={}",
                s.total_lookups,
                s.total_time_ms,
                s.only_one_hits,
                s.same_file_hits,
                s.import_hits,
                s.same_dir_hits,
                s.operand_hits,
                s.nested_var_hits,
                s.not_found
            );
        }
        drop(s);
        *stats.borrow_mut() = CallFinderStats::default();
    });
}
