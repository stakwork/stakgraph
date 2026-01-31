use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::Result;
use std::str::FromStr;

pub async fn test_java_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/java",
        Lang::from_str("java").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "java",
        "Language node name should be 'java'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/java",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 9, "Expected  files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 10, "Expected 10 directory nodes");

    let pom_file = graph.find_nodes_by_name(NodeType::File, repo.lang.kind.pkg_files()[0]);
    assert_eq!(pom_file.len(), 1, "Expected pom.xml files");
    assert_eq!(
        pom_file[0].name, "pom.xml",
        "pom.xml file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
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
    nodes_count += classes.len();
    assert_eq!(classes.len(), 5, "Expected 5 classes");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(sorted_classes[3].name, "Person", "Class name is incorrect");
    assert_eq!(
        normalize_path(&sorted_classes[3].file),
        "src/testing/java/src/main/java/graph/stakgraph/java/model/Person.java",
        "Class file path is incorrect"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 13, "Expected 13 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let requests = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 2, "Expected 2 endpoints");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;
    assert_eq!(calls_edges_count, 2, "Expected at 2 calls edges");

    let parentof = graph.count_edges_of_type(EdgeType::ParentOf);
    edges_count += parentof;
    assert_eq!(parentof, 1, "Expected at 1 parentOf edges");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges_count;
    assert_eq!(import_edges_count, 3, "Expected at 3 import edges");

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 2, "Expected 2 instances");

    let instance_edges_count = graph.count_edges_of_type(EdgeType::Of);
    edges_count += instance_edges_count;
    assert_eq!(instance_edges_count, 3, "Expected at 3 instance edges");

    let contains_edges_count = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges_count;
    assert_eq!(contains_edges_count, 49, "Expected 49 contains edges");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;
    assert_eq!(handler_edges_count, 2, "Expected at 2 handler edges");

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

    let get_person_attrs = get_person_fn.node_data.meta.get("attributes");
    assert!(
        get_person_attrs.is_some(),
        "getPerson should have attributes"
    );
    assert!(
        get_person_attrs.unwrap().contains("GetMapping"),
        "getPerson should have @GetMapping annotation"
    );
    assert_eq!(
        get_person_fn.node_data.docs,
        Some("Get a person by ID.".to_string()),
        "getPerson should have documentation"
    );

    let create_person_fn = functions
    .iter()
    .find(|f| f.name == "createPerson" && normalize_path(&f.file) == "src/testing/java/src/main/java/graph/stakgraph/java/controller/PersonController.java")
    .map(|n| Node::new(NodeType::Function, n.clone()))
    .expect("createPerson function not found");

    let create_person_attrs = create_person_fn.node_data.meta.get("attributes");
    assert!(
        create_person_attrs.is_some(),
        "createPerson should have attributes"
    );
    assert!(
        create_person_attrs.unwrap().contains("PostMapping"),
        "createPerson should have @PostMapping annotation"
    );

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

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(nodes as usize, nodes_count, "Node count mismatch");
    assert_eq!(edges as usize, edges_count, "Edge count mismatch");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_java_generic::<ArrayGraph>().await.unwrap();
    test_java_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_java_generic::<Neo4jGraph>().await.unwrap();
    }
}
