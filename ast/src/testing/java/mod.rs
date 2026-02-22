use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::Result;
use std::str::FromStr;

pub async fn test_java_generic<G: Graph + Sync>() -> Result<()> {
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
    assert_eq!(files.len(), 19, "Expected 19 Java files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 22, "Expected 22 Java directories");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(libraries.len(), 0, "Expected 0 library nodes");

    let pom_file = graph.find_nodes_by_name(NodeType::File, repo.lang.kind.pkg_files()[0]);
    assert_eq!(pom_file.len(), 1, "Expected pom.xml files");
    assert_eq!(
        pom_file[0].name, "pom.xml",
        "pom.xml file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 14, "Expected 14 import nodes");

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
    assert_eq!(classes.len(), 13, "Expected 13 class nodes");

    assert!(
        classes.iter().any(|c| c.name == "AdvancedPersonController"),
        "AdvancedPersonController class not found"
    );
    assert!(
        classes.iter().any(|c| c.name == "BillingService"),
        "BillingService class not found"
    );
    assert!(
        classes.iter().any(|c| c.name == "StripePaymentGateway"),
        "StripePaymentGateway class not found"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 26, "Expected 26 variable nodes");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 37, "Expected 37 function nodes");

    assert!(
        functions.iter().any(|f| f.name == "BillingService"),
        "BillingService constructor function not found"
    );
    assert!(
        functions.iter().any(|f| f.name == "StripePaymentGateway"),
        "StripePaymentGateway constructor function not found"
    );
    assert!(
        functions.iter().any(|f| f.name == "handleGet"),
        "handleGet function not found"
    );
    assert!(
        functions.iter().any(|f| f.name == "handlePost"),
        "handlePost function not found"
    );
    assert!(
        functions.iter().any(|f| f.name == "chargePerson"),
        "chargePerson function not found"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 2, "Expected 2 data model nodes");
    assert!(
        data_models.iter().any(|dm| dm.name == "Person"),
        "Person data model not found"
    );
    assert!(
        data_models.iter().any(|dm| dm.name == "InvoiceRecord"),
        "InvoiceRecord data model not found"
    );

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += traits.len();
    assert_eq!(traits.len(), 2, "Expected 2 trait/interface nodes");
    assert!(
        traits.iter().any(|t| t.name == "PaymentGateway"),
        "PaymentGateway trait/interface not found"
    );

    let requests = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 11, "Expected 11 endpoint nodes");

    let has_endpoint = |path: &str, verb: &str| {
        requests
            .iter()
            .any(|e| e.name == path && e.meta.get("verb") == Some(&verb.to_string()))
    };

    assert!(has_endpoint("/person/{id}", "GET"), "Missing GET /person/{{id}}");
    assert!(has_endpoint("/person", "POST"), "Missing POST /person");
    assert!(has_endpoint("/anon-get", "GET"), "Missing GET /anon-get");
    assert!(has_endpoint("/anon-post", "POST"), "Missing POST /anon-post");
    assert!(has_endpoint("/fn-get", "GET"), "Missing GET /fn-get");
    assert!(has_endpoint("/fn-post", "POST"), "Missing POST /fn-post");
    assert!(has_endpoint("/fn-put", "PUT"), "Missing PUT /fn-put");
    assert!(has_endpoint("/bulk", "POST"), "Missing POST /bulk");
    assert!(has_endpoint("/search", "GET"), "Missing GET /search");
    assert!(has_endpoint("/{id}", "DELETE"), "Missing DELETE /{{id}}");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;
    assert_eq!(calls_edges_count, 14, "Expected 14 Calls edges");

    let parentof = graph.count_edges_of_type(EdgeType::ParentOf);
    edges_count += parentof;
    assert_eq!(parentof, 1, "Expected 1 ParentOf edge");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges_count;
    assert_eq!(import_edges_count, 9, "Expected 9 Imports edges");

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 7, "Expected 7 instance nodes");
    assert!(
        instances.iter().any(|i| i.name == "billingService"),
        "billingService instance not found"
    );

    let instance_edges_count = graph.count_edges_of_type(EdgeType::Of);
    edges_count += instance_edges_count;
    assert_eq!(instance_edges_count, 8, "Expected 8 Of edges");

    let contains_edges_count = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges_count;
    assert_eq!(contains_edges_count, 196, "Expected 196 Contains edges");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;
    assert_eq!(handler_edges_count, 11, "Expected 11 Handler edges");

    let nested_in_edges_count = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in_edges_count;
    assert_eq!(nested_in_edges_count, 3, "Expected 3 NestedIn edges");

    let impl_edges_count = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += impl_edges_count;
    assert_eq!(impl_edges_count, 1, "Expected 1 Implements edge");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += unit_tests.len();
    nodes_count += integration_tests.len();
    assert_eq!(unit_tests.len(), 1, "Expected 1 unit test node");
    assert_eq!(integration_tests.len(), 1, "Expected 1 integration test node");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 0, "Expected 0 e2e test nodes");

    let request_nodes = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += request_nodes.len();
    assert_eq!(request_nodes.len(), 0, "Expected 0 request nodes");

    let package_nodes = graph.find_nodes_by_type(NodeType::Package);
    nodes_count += package_nodes.len();
    assert_eq!(package_nodes.len(), 0, "Expected 0 package nodes");

    let operand_edges = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges;
    assert_eq!(operand_edges, 0, "Expected 0 Operand edges");

    let uses_edges = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses_edges;
    assert_eq!(uses_edges, 0, "Expected 0 Uses edges");

    let argof_edges = graph.count_edges_of_type(EdgeType::ArgOf);
    edges_count += argof_edges;
    assert_eq!(argof_edges, 0, "Expected 0 ArgOf edges");

    let includes_edges = graph.count_edges_of_type(EdgeType::Includes);
    edges_count += includes_edges;
    assert_eq!(includes_edges, 0, "Expected 0 Includes edges");

    let renders_edges = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders_edges;
    assert_eq!(renders_edges, 0, "Expected 0 Renders edges");
    assert!(unit_tests.len() >= 1, "Expected unit test nodes");
    assert!(integration_tests.len() >= 1, "Expected integration test nodes");

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

    let fn_get_endpoint = requests
        .iter()
        .find(|e| e.name == "/fn-get" && e.meta.get("verb") == Some(&"GET".to_string()))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /fn-get endpoint not found");

    let fn_get_handler = functions
        .iter()
        .find(|f| f.name == "handleGet")
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("handleGet function not found");

    assert!(
        graph.has_edge(&fn_get_endpoint, &fn_get_handler, EdgeType::Handler),
        "Expected /fn-get to be handled by handleGet method reference"
    );

    let trait_gateway = traits
        .iter()
        .find(|t| t.name == "PaymentGateway")
        .map(|n| Node::new(NodeType::Trait, n.clone()))
        .expect("PaymentGateway trait not found");

    let stripe_impl = classes
        .iter()
        .find(|c| c.name == "StripePaymentGateway")
        .map(|n| Node::new(NodeType::Class, n.clone()))
        .expect("StripePaymentGateway class not found");

    assert!(
        graph.has_edge(&stripe_impl, &trait_gateway, EdgeType::Implements),
        "Expected StripePaymentGateway to implement PaymentGateway"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(nodes as usize, nodes_count, "Node count mismatch");
    assert_eq!(edges as usize, edges_count, "Edge count mismatch");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_java_generic::<ArrayGraph>().await.unwrap();
        test_java_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_java_generic::<Neo4jGraph>().await.unwrap();
    }
}
