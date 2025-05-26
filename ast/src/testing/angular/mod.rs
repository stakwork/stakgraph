use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use anyhow::Ok;
use std::str::FromStr;

pub async fn test_angular_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/angular",
        Lang::from_str("angular").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let (num_nodes, num_edges) = graph.get_graph_size().await;
    assert_eq!(num_nodes, 76, "Expected 76 nodes");
    assert_eq!(num_edges, 78, "Expected 78 edges");

    let imports = graph.find_nodes_by_type(NodeType::Import).await;
    assert_eq!(imports.len(), 10, "Expected 10 imports");

    let classes = graph.find_nodes_by_type(NodeType::Class).await;
    assert_eq!(classes.len(), 5, "Expected 5 classes");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel).await;
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function).await;
    assert_eq!(functions.len(), 8, "Expected 8 functions");

    // Check if there's a constructor function
    let constructor = functions.iter().find(|f| f.name == "constructor");
    assert!(
        constructor.is_some(),
        "Expected constructor function not found"
    );

    // Test requests
    let requests = graph.find_nodes_by_type(NodeType::Request).await;
    assert_eq!(requests.len(), 7, "Expected 7 requests");

    // Test calls edges
    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls).await;
    assert_eq!(calls_edges_count, 8, "Expected 8 calls edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_angular() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_angular_generic::<ArrayGraph>().await.unwrap();
    test_angular_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_angular_generic::<Neo4jGraph>().await.unwrap();
    }
}
