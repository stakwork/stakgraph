use console::style;
use shared::Result;

use crate::output::Output;

use super::connection::connect_graph_ops;
use super::{ALL_EDGE_TYPES, ALL_NODE_TYPES};

pub(super) fn run_schema(out: &mut Output) -> Result<()> {
    out.writeln(style("Node types:").bold().to_string())?;
    for nt in ALL_NODE_TYPES {
        out.writeln(format!("  {}", nt))?;
    }
    out.newline()?;
    out.writeln(style("Edge types:").bold().to_string())?;
    for et in ALL_EDGE_TYPES {
        out.writeln(format!("  {:?}", et))?;
    }
    Ok(())
}

pub(super) async fn run_clear(out: &mut Output) -> Result<()> {
    use std::io::{self, BufRead, Write};

    print!("This will delete all nodes and edges from the graph. Continue? [y/N] ");
    io::stdout().flush().ok();

    let stdin = io::stdin();
    let line = stdin.lock().lines().next().unwrap_or(Ok(String::new()))?;
    if line.trim().to_lowercase() != "y" {
        out.writeln("Aborted.".to_string())?;
        return Ok(());
    }

    let mut ops = connect_graph_ops().await?;
    let spinner = crate::progress::CliSpinner::new("Clearing graph...");
    ops.clear().await?;
    spinner.finish_with_message("Graph cleared");
    let (nodes, edges) = ops.get_graph_size().await?;
    out.writeln(format!("Graph now: {} nodes, {} edges", nodes, edges))?;
    Ok(())
}

pub(super) async fn run_stats(out: &mut Output) -> Result<()> {
    let ops = connect_graph_ops().await?;
    let (nodes, edges) = ops.get_graph_size().await?;
    out.writeln(format!("Nodes: {}", nodes))?;
    out.writeln(format!("Edges: {}", edges))?;
    Ok(())
}
