use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
// use crate::utils::get_use_lsp;
use crate::{
    lang::Lang,
    repo::{Repo, Repos},
};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_php_generic<G: Graph>() -> Result<()> {
    let use_lsp = false;
    let repo = Repo::new(
        "src/testing/php",
        Lang::from_str("php").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let repos = Repos(vec![repo]);
    let graph = repos.build_graphs_inner::<G>().await?;

    graph.analysis();

    let mut nodes = 0;
    let mut edges = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(language_nodes[0].name, "php");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes += file_nodes.len();
    assert_eq!(file_nodes.len(), 15, "Expected 15 File nodes");

    let _web_routes = file_nodes
        .iter()
        .find(|f| f.file.ends_with("php/routes/web.php"))
        .expect("routes/web.php not found");

    let _api_routes = file_nodes
        .iter()
        .find(|f| f.file.ends_with("php/routes/api.php"))
        .expect("routes/api.php not found");

    let user_controller_file = file_nodes
        .iter()
        .find(|f| {
            f.file
                .ends_with("php/app/Http/Controllers/UserController.php")
        })
        .expect("UserController.php not found");

    let directory_nodes = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directory_nodes.len();

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 9, "Expected 9 Class nodes");

    let _user_class = classes
        .iter()
        .find(|c| c.name == "User")
        .expect("User model class not found");

    let user_controller_class = classes
        .iter()
        .find(|c| c.name == "UserController")
        .expect("UserController class not found");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    assert_eq!(functions.len(), 19, "Expected 19 Function nodes");

    let index_method = functions
        .iter()
        .find(|f| f.name == "index" && f.file.ends_with("UserController.php"))
        .expect("UserController::index not found");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 38, "Expected 38 Endpoint nodes");

    let _root_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/" && e.file.ends_with("web.php"))
        .expect("Root endpoint '/' in web.php not found");

    let _dashboard_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/dashboard" && e.file.ends_with("web.php"))
        .expect("Dashboard endpoint in web.php not found");

    let _login_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/login" && e.file.ends_with("api.php"))
        .expect("Login endpoint in api.php not found");

    let _blog_index = endpoints
        .iter()
        .find(|e| (e.name == "/" || e.name == "/blog") && e.file.ends_with("BlogController.php"))
        .expect("BlogController endpoint not found");
    let _group_index = endpoints
        .iter()
        .find(|e| e.name == "/users/group_index" && e.file.ends_with("web_groups.php"))
        .expect("Group index endpoint not found");

    let _group_store = endpoints
        .iter()
        .find(|e| e.name == "/users/group_store" && e.file.ends_with("web_groups.php"))
        .expect("Group store endpoint not found");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 11, "Expected 11 Import nodes");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 11, "Expected 11 Library nodes");

    let tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes += tests.len();
    assert_eq!(tests.len(), 3, "Expected 3 UnitTest nodes");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes += integration_tests.len();
    assert_eq!(
        integration_tests.len(),
        3,
        "Expected 3 IntegrationTest nodes"
    );

    let vars = graph.find_nodes_by_type(NodeType::Var);
    nodes += vars.len();
    assert_eq!(vars.len(), 7, "Expected 7 Var nodes");

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes += traits.len();
    assert_eq!(traits.len(), 0, "Expected 0 Trait nodes");

    let datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += datamodels.len();
    assert_eq!(datamodels.len(), 0, "Expected 0 DataModel nodes");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges += operands;
    assert_eq!(operands, 14, "Expected 14 Operand edges");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;
    assert_eq!(calls, 6, "Expected 6 Calls edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 92, "Expected 92 Contains edges");

    let parent_of = graph.count_edges_of_type(EdgeType::ParentOf);
    edges += parent_of;
    assert_eq!(parent_of, 1, "Expected 1 ParentOf edges");

    assert!(
        graph.has_edge(
            &Node::new(NodeType::File, user_controller_file.clone()),
            &Node::new(NodeType::Class, user_controller_class.clone()),
            EdgeType::Contains
        ),
        "UserController.php should contain UserController class"
    );

    assert!(
        graph.has_edge(
            &Node::new(NodeType::Class, user_controller_class.clone()),
            &Node::new(NodeType::Function, index_method.clone()),
            EdgeType::Operand
        ),
        "UserController class should have index method (Operand edge)"
    );

    let imports_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges += imports_edges;
    assert_eq!(
        imports_edges, 0,
        "Expected 0 Imports edges (not linked yet)"
    );

    // Handler edges (link endpoints to functions)
    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges += handlers;
    assert_eq!(handlers, 8, "Expected 8 Handler edges");

    let (num_nodes, num_edges) = graph.get_graph_size();

    assert_eq!(
        num_nodes, nodes as u32,
        "Nodes mismatch: expected {num_nodes} nodes found {nodes}"
    );

    assert_eq!(
        num_edges, edges as u32,
        "Edges mismatch: expected {edges} edges found {num_edges}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_php() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_php_generic::<ArrayGraph>().await.unwrap();
    test_php_generic::<BTreeMapGraph>().await.unwrap();
}
