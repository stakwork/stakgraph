mod connection;
mod ingest;
mod map;
mod node;
mod parse_helpers;
mod schema;
mod search;

use ast::lang::graphs::{EdgeType, NodeType};
use shared::Result;

use crate::args::{GraphArgs, GraphCommand};
use crate::output::Output;

pub use parse_helpers::print_caller_counts;

pub(super) const ALL_NODE_TYPES: &[NodeType] = &[
    NodeType::Repository,
    NodeType::Package,
    NodeType::Language,
    NodeType::Directory,
    NodeType::File,
    NodeType::Import,
    NodeType::Library,
    NodeType::Class,
    NodeType::Trait,
    NodeType::Instance,
    NodeType::Function,
    NodeType::Endpoint,
    NodeType::Request,
    NodeType::DataModel,
    NodeType::Feature,
    NodeType::Page,
    NodeType::Var,
    NodeType::UnitTest,
    NodeType::IntegrationTest,
    NodeType::E2eTest,
];

pub(super) const ALL_EDGE_TYPES: &[EdgeType] = &[
    EdgeType::Calls,
    EdgeType::Uses,
    EdgeType::Operand,
    EdgeType::ArgOf,
    EdgeType::Contains,
    EdgeType::Imports,
    EdgeType::Of,
    EdgeType::Handler,
    EdgeType::Includes,
    EdgeType::Renders,
    EdgeType::ParentOf,
    EdgeType::Implements,
    EdgeType::NestedIn,
];

pub async fn run_graph(args: &GraphArgs, out: &mut Output) -> Result<()> {
    match &args.command {
        GraphCommand::Ingest(a) => ingest::run_ingest(&a.path, out).await,
        GraphCommand::Search(a) => search::run_search(&a.query, &a.node_type, a.limit, out).await,
        GraphCommand::Node(a) => node::run_node(&a.name, out).await,
        GraphCommand::Map(a) => {
            map::run_map(&a.name, a.node_type.as_deref(), &a.direction, a.depth, a.tests, &a.trim, out).await
        }
        GraphCommand::Schema => schema::run_schema(out),
        GraphCommand::Clear => schema::run_clear(out).await,
        GraphCommand::Stats => schema::run_stats(out).await,
    }
}
