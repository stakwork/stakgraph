use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{lang::Lang, repo::Repo};
use anyhow::Result;
use std::str::FromStr;

pub async fn test_java_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/java",
        Lang::from_str("java").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let (num_nodes, num_edges) = graph.get_graph_size();

    //graph.analysis();
    assert_eq!(num_nodes, 38, "Expected 38 nodes");
    assert_eq!(num_edges, 45, "Expected 45 edges");

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "java",
        "Language node name should be 'java'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/java/",
        "Language node file path is incorrect"
    );

    let pom_file = graph.find_nodes_by_name(NodeType::File, repo.lang.kind.pkg_files()[0]);
    assert_eq!(pom_file.len(), 1, "Expected pom.xml files");
    assert_eq!(
        pom_file[0].name, "pom.xml",
        "pom.xml file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 4, "Expected 4 imports");

    let main_import_body = format!(
        r#"package graph.stakgraph.java.controller;

import graph.stakgraph.java.model.Person;
import graph.stakgraph.java.repository.PersonRepository;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import java.util.Optional;"#
    );
    let main = imports
        .iter()
        .find(|i| {
            i.file
                == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java"
        })
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 3, "Expected 3 classes");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    for imp in &imports {
        println!("Import: {:?}\n\n", imp);
    }
    assert_eq!(imports.len(), 4, "Expected 4 imports");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(sorted_classes[1].name, "Person", "Class name is incorrect");
    assert_eq!(
        normalize_path(&sorted_classes[1].file),
        "src/testing/java/src/main/java/graph/stakgraph/java/model/Person.java",
        "Class file path is incorrect"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 11, "Expected 11 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let requests = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(requests.len(), 2, "Expected 2 endpoints");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges_count, 2, "Expected at 2 calls edges");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    assert_eq!(import_edges_count, 2, "Expected at 2 import edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
    #[cfg(feature = "neo4j")]
    use crate::lang::graphs::Neo4jGraph;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_java_generic::<ArrayGraph>().await.unwrap();
    test_java_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();
        test_java_generic::<Neo4jGraph>().await.unwrap();
    }
}
