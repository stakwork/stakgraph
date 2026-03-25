use ast::lang::graphs::graph_ops::GraphOps;
use shared::Result;

pub(super) async fn connect_graph_ops() -> Result<GraphOps> {
    let mut ops = GraphOps::new();
    ops.connect().await?;
    Ok(ops)
}
