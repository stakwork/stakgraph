use crate::lang::graphs::{ArrayGraph, BTreeMapGraph, EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_c_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/c",
        Lang::from_str("c").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 Language node");
    assert_eq!(language_nodes[0].name, "c");

    let repo_nodes = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repo_nodes.len();
    assert_eq!(repo_nodes.len(), 1, "Expected 1 Repository node");

    let dir_nodes = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += dir_nodes.len();
    assert_eq!(dir_nodes.len(), 6, "Expected 6 Directory nodes");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes_count += file_nodes.len();
    assert_eq!(file_nodes.len(), 37, "Expected 37 File nodes");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 23, "Expected 23 Class nodes");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 46, "Expected 46 Function nodes");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 4, "Expected 4 Endpoint nodes");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 31, "Expected 31 Import nodes");

    let packages = graph.find_nodes_by_type(NodeType::Package);
    nodes_count += packages.len();
    assert_eq!(packages.len(), 0, "Expected 0 Package nodes");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += traits.len();

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();

    let repo_node = Node::new(NodeType::Repository, repo_nodes[0].clone());

    let embedded_dir_data = dir_nodes
        .iter()
        .find(|d| d.name == "embedded-iot")
        .expect("embedded-iot dir not found");
    let embedded_dir_node = Node::new(NodeType::Directory, embedded_dir_data.clone());

    assert!(
        graph.has_edge(&repo_node, &embedded_dir_node, EdgeType::Contains),
        "Repo should contain embedded-iot"
    );

    let protocols_dir_data = dir_nodes
        .iter()
        .find(|d| d.name == "protocols")
        .expect("protocols dir not found");
    let protocols_dir_node = Node::new(NodeType::Directory, protocols_dir_data.clone());

    assert!(
        graph.has_edge(&embedded_dir_node, &protocols_dir_node, EdgeType::Contains),
        "embedded-iot should contain protocols"
    );

    let sensors_c_data = file_nodes
        .iter()
        .find(|f| f.name == "sensors.c")
        .expect("sensors.c not found");
    let sensors_c_node = Node::new(NodeType::File, sensors_c_data.clone());

    assert!(
        graph.has_edge(&embedded_dir_node, &sensors_c_node, EdgeType::Contains),
        "embedded-iot should contain sensors.c"
    );

    let user_struct = classes
        .iter()
        .find(|c| c.name == "User")
        .expect("User struct not found");
    if let Some(docs) = &user_struct.docs {
        if !docs.contains("Represents a user") {
            eprintln!("WARNING: User struct docs content mismatch");
        }
    }

    let create_user_fn = functions
        .iter()
        .find(|f| f.name == "create_user")
        .expect("create_user function not found");
    if let Some(docs) = &create_user_fn.docs {
        if !docs.contains("Creates a new user") {
            eprintln!("WARNING: create_user docs content mismatch");
        }
    }
    assert!(create_user_fn
        .file
        .ends_with("src/testing/c/web-http/models.c"));

    let create_product_fn = functions
        .iter()
        .find(|f| f.name == "create_product")
        .expect("create_product function not found");
    if let Some(docs) = &create_product_fn.docs {
        if !docs.contains("Creates a new product") {
            eprintln!("WARNING: create_product docs content mismatch");
        }
    }

    let temp_ops = classes
        .iter()
        .find(|c| c.name == "SensorOps")
        .expect("SensorOps struct not found");
    if let Some(docs) = &temp_ops.docs {
        if !docs.contains("Virtual table for sensor") {
            eprintln!("WARNING: SensorOps docs mismatch");
        }
    }

    let user_endpoint = endpoints
        .iter()
        .find(|e| e.name.contains("users/([0-9]+)"))
        .expect("User endpoint not found");
    let handler_get_user = functions
        .iter()
        .find(|f| f.name == "handler_get_user")
        .expect("handler_get_user not found");

    let user_endpoint_node = Node::new(NodeType::Endpoint, user_endpoint.clone());
    let handler_get_user_node = Node::new(NodeType::Function, handler_get_user.clone());
    assert!(
        graph.has_edge(
            &user_endpoint_node,
            &handler_get_user_node,
            EdgeType::Handler
        ),
        "Endpoint -> Handler edge missing for users/([0-9]+)"
    );

    let sensors_h_import = imports
        .iter()
        .find(|i| i.file.ends_with("sensors.c") && i.body.contains("sensors.h"))
        .expect("sensors.h import in sensors.c not found");
    let sensors_h_import_node = Node::new(NodeType::Import, sensors_h_import.clone());

    assert!(
        graph.has_edge(&sensors_c_node, &sensors_h_import_node, EdgeType::Contains),
        "sensors.c should contain sensors.h import"
    );

    let safe_strdup = functions
        .iter()
        .find(|f| f.name == "safe_strdup")
        .expect("safe_strdup not found");
    assert!(safe_strdup.file.ends_with("lib/string_utils.c"));

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges;

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges;

    let import_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges;

    let handler_edges = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges;

    let uses_edges = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses_edges;

    let operand_edges = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges;

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;

    let nested_in_edges = graph.count_edges_of_type(EdgeType::NestedIn);
    edges_count += nested_in_edges;

    let renders_edges = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders_edges;

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes got {}",
        nodes, nodes_count
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected {} edges got {}",
        edges, edges_count
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_c() {
    test_c_generic::<ArrayGraph>().await.unwrap();
    test_c_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_c_generic::<Neo4jGraph>().await.unwrap();
    }
}
