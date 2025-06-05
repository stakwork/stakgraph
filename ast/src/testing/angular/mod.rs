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

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 80, "Expected 80 nodes");
    assert_eq!(num_edges, 90, "Expected 90 edges");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 10, "Expected 10 imports");

    let main_import_body = format!(
        r#"import {{ bootstrapApplication }} from '@angular/platform-browser';
import {{ appConfig }} from './app/app.config';
import {{ AppComponent }} from './app/app.component';"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/angular/src/main.ts")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 5, "Expected 5 classes");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 8, "Expected 8 functions");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 4, "Expected 4 variables");

    let constructor = functions.iter().find(|f| f.name == "constructor");
    assert!(
        constructor.is_some(),
        "Expected constructor function not found"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 7, "Expected 7 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges_count, 8, "Expected 8 calls edges");

    let imports_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    assert_eq!(imports_edges_count, 8, "Expected 8 imports edges");

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
