use crate::lang::graphs::{ArrayGraph, BTreeMapGraph, EdgeType, NodeType};
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
    assert_eq!(files.len(), 10, "Expected 10 files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 13, "Expected 13 directory nodes");

    let pom_file = graph.find_nodes_by_name(NodeType::File, repo.lang.kind.pkg_files()[0]);
    assert_eq!(pom_file.len(), 1, "Expected pom.xml files");
    assert_eq!(
        pom_file[0].name, "pom.xml",
        "pom.xml file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 5, "Expected 5 imports");

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
    assert_eq!(classes.len(), 6, "Expected 6 classes");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 16, "Expected 16 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let requests = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 4, "Expected 4 endpoints");

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

    assert_eq!(contains_edges_count, 58, "Expected 58 contains edges");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;
    assert_eq!(handler_edges_count, 4, "Expected at 4 handler edges");

    let nested_in_edges_count = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in_edges_count;
    assert_eq!(nested_in_edges_count, 2, "Expected 2 NestedIn edges");

    let anon_get_endpoint = requests
        .iter()
        .find(|e| e.name == "/anon-get")
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("Anonymous GET endpoint not found");

    let anon_post_endpoint = requests
        .iter()
        .find(|e| e.name == "/anon-post")
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("Anonymous POST endpoint not found");

    let anon_get_handler = functions
        .iter()
        .find(|f| f.name.contains("GET_anon_get_lambda"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Anonymous GET lambda handler not found");

    let anon_post_handler = functions
        .iter()
        .find(|f| f.name.contains("POST_anon_post_lambda"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Anonymous POST lambda handler not found");

    assert!(
        graph.has_edge(&anon_get_endpoint, &anon_get_handler, EdgeType::Handler),
        "Expected /anon-get to be handled by lambda"
    );

    assert!(
        graph.has_edge(&anon_post_endpoint, &anon_post_handler, EdgeType::Handler),
        "Expected /anon-post to be handled by lambda"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(nodes as usize, nodes_count, "Node count mismatch");
    assert_eq!(edges as usize, edges_count, "Edge count mismatch");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
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
