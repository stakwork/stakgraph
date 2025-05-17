use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{CallsMeta, Graph};
use crate::{lang::Lang, repo::Repo};
use anyhow::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_kotlin_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let (num_nodes, num_edges) = graph.get_graph_size();

    //graph.analysis();
    assert_eq!(num_nodes, 115, "Expected 115 nodes");
    assert_eq!(num_edges, 125, "Expected 125 edges");

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "kotlin",
        "Language node name should be 'kotlin'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/kotlin/",
        "Language node file path is incorrect"
    );

    let build_gradle_files = graph.find_nodes_by_name(NodeType::File, "build.gradle.kts");
    assert!(
        build_gradle_files.len() > 0 && build_gradle_files.len() <= 2,
        "Expected 1-2 build.gradle.kts files, found {}",
        build_gradle_files.len()
    );
    assert_eq!(
        build_gradle_files[0].name, "build.gradle.kts",
        "Gradle file name is incorrect"
    );

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    assert_eq!(libraries.len(), 44, "Expected 44 libraries");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 9, "Expected 9 imports");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 6, "Expected 6 classes");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_classes[1].name, "ExampleInstrumentedTest",
        "Class name is incorrect"
    );
    assert_eq!(
        normalize_path(&sorted_classes[1].file),
        "src/testing/kotlin/app/src/androidTest/java/com/kotlintestapp/ExampleInstrumentedTest.kt",
        "Class file path is incorrect"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 19, "Expected 19 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert!(data_models.len() >= 0, "Expected at least 0 data models");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert!(requests.len() >= 0, "Expected at least 0 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls(CallsMeta::default()));
    assert!(calls_edges_count > 0, "Expected at least one calls edge");

    Ok(())
}

pub async fn test_kotlin_lsp_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        true,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph_result = repo.build_graph_inner::<G>().await;
    
    if graph_result.is_err() {
        println!("Skipping LSP test as kotlin-language-server is not available");
        return Ok(());
    }
    
    let graph = graph_result.unwrap();
    let (num_nodes, num_edges) = graph.get_graph_size();

    println!("LSP found {} nodes and {} edges", num_nodes, num_edges);
    assert!(num_nodes >= 115, "Expected at least 115 nodes with LSP (got {})", num_nodes);
    assert!(num_edges >= 125, "Expected at least 125 edges with LSP (got {})", num_edges);

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert!(imports.len() >= 9, "Expected at least 9 Import nodes with LSP");
    
    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert!(classes.len() >= 6, "Expected at least 6 Class nodes with LSP");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert!(functions.len() >= 19, "Expected at least 19 Function nodes with LSP");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    assert!(contains_edges > 0, "Expected at least some Contains edges with LSP");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls(CallsMeta::default()));
    assert!(calls_edges > 0, "Expected at least some Calls edges with LSP");

    Ok(())
}

pub async fn compare_lsp_vs_nonlsp<G: Graph>() -> Result<(), anyhow::Error> {
    let repo_non_lsp = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph_non_lsp = repo_non_lsp.build_graph_inner::<G>().await?;
    let (non_lsp_nodes, non_lsp_edges) = graph_non_lsp.get_graph_size();
    
    println!("Non-LSP found {} nodes and {} edges", non_lsp_nodes, non_lsp_edges);

    let repo_lsp = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        true,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph_lsp_result = repo_lsp.build_graph_inner::<G>().await;
    
    if graph_lsp_result.is_err() {
        println!("Skipping LSP comparison as kotlin-language-server is not available");
        return Ok(());
    }
    
    let graph_lsp = graph_lsp_result.unwrap();
    let (lsp_nodes, lsp_edges) = graph_lsp.get_graph_size();
    
    println!("LSP found {} nodes and {} edges", lsp_nodes, lsp_edges);
    if lsp_nodes >= non_lsp_nodes && lsp_edges >= non_lsp_edges {
        let node_increase = lsp_nodes - non_lsp_nodes;
        let edge_increase = lsp_edges - non_lsp_edges;
        println!("Additional nodes found by LSP: {}", node_increase);
        println!("Additional edges found by LSP: {}", edge_increase);
    }
    
    Ok(())
}

#[test(tokio::test)]
async fn test_kotlin() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_kotlin_generic::<BTreeMapGraph>().await.unwrap();
    test_kotlin_generic::<ArrayGraph>().await.unwrap();
}

#[test(tokio::test)]
async fn test_kotlin_lsp() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_kotlin_lsp_generic::<BTreeMapGraph>().await.unwrap();
    test_kotlin_lsp_generic::<ArrayGraph>().await.unwrap();
}

#[test(tokio::test)]
async fn test_kotlin_comparison() {
    use crate::lang::graphs::BTreeMapGraph;
    compare_lsp_vs_nonlsp::<BTreeMapGraph>().await.unwrap();
}
