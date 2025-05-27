use crate::lang::graphs::{EdgeType, NodeType};
use crate::utils::get_use_lsp;
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use anyhow::Result;
use std::str::FromStr;

pub async fn test_kotlin_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph_result = repo.build_graph_inner::<G>().await;
    
    let (graph, effective_lsp) = match (use_lsp, graph_result) {
        (true, Ok(g)) => (g, true), 
        (true, Err(_)) => {
            let non_lsp_repo = Repo::new(
                "src/testing/kotlin",
                Lang::from_str("kotlin").unwrap(),
                false,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            (non_lsp_repo.build_graph_inner::<G>().await?, false)
        },
        (false, Ok(g)) => (g, false),
        (false, Err(e)) => return Err(e),
    };

    let (num_nodes, num_edges) = graph.get_graph_size();

    if effective_lsp {
        println!("Testing with LSP: found {} nodes and {} edges", num_nodes, num_edges);
        assert_eq!(num_nodes, 120, "Expected 120 nodes with LSP");
        assert_eq!(num_edges, 135, "Expected 135 edges with LSP");
    } else {
        println!("Testing without LSP: found {} nodes and {} edges", num_nodes, num_edges);
        assert_eq!(num_nodes, 115, "Expected 115 nodes");
        assert_eq!(num_edges, 125, "Expected 125 edges");
    }

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
    if effective_lsp {
        assert_eq!(
            build_gradle_files.len(),
            2,
            "Expected 2 build.gradle.kts files with LSP"
        );
    } else {
        assert_eq!(
            build_gradle_files.len(),
            2,
            "Expected 2 build.gradle.kts files"
        );
    }
    assert_eq!(
        build_gradle_files[0].name, "build.gradle.kts",
        "Gradle file name is incorrect"
    );

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    if effective_lsp {
        assert_eq!(libraries.len(), 48, "Expected 48 libraries with LSP");
    } else {
        assert_eq!(libraries.len(), 44, "Expected 44 libraries");
    }

    let imports = graph.find_nodes_by_type(NodeType::Import);
    if effective_lsp {
        assert_eq!(imports.len(), 11, "Expected 11 imports with LSP");
    } else {
        assert_eq!(imports.len(), 9, "Expected 9 imports");
    }

    let classes = graph.find_nodes_by_type(NodeType::Class);
    if effective_lsp {
        assert_eq!(classes.len(), 8, "Expected 8 classes with LSP");
    } else {
        assert_eq!(classes.len(), 6, "Expected 6 classes");
    }

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
    if effective_lsp {
        assert_eq!(functions.len(), 23, "Expected 23 functions with LSP");
    } else {
        assert_eq!(functions.len(), 19, "Expected 19 functions");
    }

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    if effective_lsp {
        assert_eq!(calls_edges_count, 12, "Expected 12 calls edges with LSP");
    } else {
        assert_eq!(calls_edges_count, 13, "Expected 13 calls edges");
    }
  
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_kotlin() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_kotlin_generic::<BTreeMapGraph>().await.unwrap();
    test_kotlin_generic::<ArrayGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_kotlin_generic::<Neo4jGraph>().await.unwrap();
    }
}
