use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use anyhow::Result;
use std::str::FromStr;
use test_log::test;

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

    graph.analysis();

    assert_eq!(num_nodes, 42, "Expected 42 nodes");
    assert_eq!(num_edges, 49, "Expected 49 edges");

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

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    assert_eq!(instances.len(), 0, "Expected 0 instances");

    let person_class = classes
        .iter()
        .find(|c| {
            c.name == "Person"
                && normalize_path(&c.file)
                    == "src/testing/java/src/main/java/graph/stakgraph/java/model/Person.java"
        })
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("Person class not found");

    let person_data_model = data_models
        .iter()
        .find(|dm| {
            dm.name == "Person"
                && normalize_path(&dm.file)
                    == "src/testing/java/src/main/java/graph/stakgraph/java/model/Person.java"
        })
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Person DataModel not found");

    let get_person_endpoint = requests
    .iter()
    .find(|e| e.name == "/person/{id}" && normalize_path(&e.file) == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java")
    .map(|n| Node::new(NodeType::Endpoint, n.clone()))
    .expect("GET /person/{id} endpoint not found");

    let post_person_endpoint = requests
    .iter()
    .find(|e| e.name == "/person" && normalize_path(&e.file) == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java")
    .map(|n| Node::new(NodeType::Endpoint, n.clone()))
    .expect("POST /person endpoint not found");

    let get_person_fn = functions
    .iter()
    .find(|f| f.name == "getPerson" && normalize_path(&f.file) == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java")
    .map(|n| Node::new(NodeType::Function, n.clone()))
    .expect("getPerson function not found");

    let create_person_fn = functions
    .iter()
    .find(|f| f.name == "createPerson" && normalize_path(&f.file) == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java")
    .map(|n| Node::new(NodeType::Function, n.clone()))
    .expect("createPerson function not found");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/id' endpoint to be handled by getPerson"
    );

    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' endpoint to be handled by createPerson"
    );

    let person_model_file = graph
        .find_nodes_by_name(NodeType::File, "Person.java")
        .into_iter()
        .find(|n| {
            normalize_path(&n.file)
                == "src/testing/java/src/main/java/graph/stakgraph/java/model/Person.java"
        })
        .map(|n| crate::lang::Node::new(NodeType::File, n))
        .expect("Person.java file node not found");

    assert!(
        graph.has_edge(&person_model_file, &person_class, EdgeType::Contains),
        "Expected Person.java file to contain Person class"
    );

    assert!(
        graph.has_edge(&person_model_file, &person_data_model, EdgeType::Contains),
        "Expected Person.java file to contain Person DataModel"
    );
    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_java() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_java_generic::<ArrayGraph>().await.unwrap();
    test_java_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let mut graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_java_generic::<Neo4jGraph>().await.unwrap();
    }
}
