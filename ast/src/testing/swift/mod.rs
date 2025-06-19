use crate::lang::graphs::NodeType;
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

pub async fn test_swift_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/swift",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let (num_nodes, num_edges) = graph.get_graph_size();

    graph.analysis();

    assert_eq!(num_nodes, 57, "Expected 57 nodes");
    assert_eq!(num_edges, 83, "Expected 83 edges");

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "swift",
        "Language node name should be 'swift'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/swift/",
        "Language node file path is incorrect"
    );

    let files = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(files.len(), 8, "Expected 8 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 7, "Expected 7 imports");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 7, "Expected 7 classes");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_classes[0].name, "API",
        "First class name should be 'API'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 26, "Expected 26 functions");

    let mut sorted_functions = functions.clone();
    sorted_functions.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_functions[0].name, "application",
        "First function name should be 'application'"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 2, "Expected 2 variables");

    let mut sorted_requests = requests.clone();
    sorted_requests.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_requests[0].name, "/people",
        "First request URL should be '/people'"
    );

    let calls = graph.count_edges_of_type(crate::lang::EdgeType::Calls);
    assert_eq!(calls, 2, "Expected 2 call edges");

    let contains = graph.count_edges_of_type(crate::lang::EdgeType::Contains);
    assert_eq!(contains, 55, "Expected 55 contains edges");

    let operands = graph.count_edges_of_type(crate::lang::EdgeType::Operand);
    assert_eq!(operands, 26, "Expected 26 operand edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_swift() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_swift_generic::<ArrayGraph>().await.unwrap();
    test_swift_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let mut graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_swift_generic::<Neo4jGraph>().await.unwrap();
    }
}
