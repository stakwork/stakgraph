use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_svelte_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/svelte",
        Lang::from_str("svelte").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;
    let (num_nodes, num_edges) = graph.get_graph_size();

    graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "svelte",
        "Language node name should be 'svelte'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/svelte",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 14, "Expected 14 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 14, "Expected 14 imports");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 4, "Expected 4 classes extracted from script blocks");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 13, "Expected 13 functions extracted from script blocks");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 0, "Expected 0 unit tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(integration_tests.len(), 0, "Expected 0 integration tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 0, "Expected 0 e2e tests");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 26, "Expected 26 data models");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 7, "Expected 7 directories");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 46, "Expected 46 request nodes from TypeScript extraction");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 0, "Expected 0 variables");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls;
    assert_eq!(calls, 52, "Expected 52 call edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 78, "Expected 78 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 of edge");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operands;
    assert_eq!(operands, 0, "Expected 0 operand edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    assert_eq!(handlers, 0, "Expected 0 handler edges");

    let imports_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imports_edges;
    assert_eq!(imports_edges, 0, "Expected 0 import edges");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    assert_eq!(uses, 0, "Expected 0 uses edges");

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in;
    assert_eq!(nested_in, 0, "Expected 0 nested in edges");

    assert_eq!(
        num_nodes as usize, nodes_count,
        "Expected {} total nodes", nodes_count
    );
    assert_eq!(
        num_edges as usize, edges_count,
        "Expected {} total edges", edges_count
    );

    Ok(())
}

#[test(tokio::test)]
async fn test_svelte() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_svelte_generic::<ArrayGraph>().await.unwrap();
    test_svelte_generic::<BTreeMapGraph>().await.unwrap();
}
