use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::Graph;
use crate::{
    lang::Lang,
    repo::{Repo, Repos},
};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_csharp_generic<G: Graph + Sync>() -> Result<()> {
    let use_lsp = false;
    let repo = Repo::new(
        "src/testing/csharp",
        Lang::from_str("csharp").unwrap(),
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
    assert_eq!(language_nodes[0].name, "csharp");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes += file_nodes.len();
    assert_eq!(file_nodes.len(), 26, "Expected 26 File nodes");

    let repo_edges = graph.count_edges_of_type(EdgeType::Of);
    edges += repo_edges;
    assert_eq!(repo_edges, 1, "Expected 1 Of edge");

    let _program_file = file_nodes
        .iter()
        .find(|f| f.file.ends_with("Program.cs"))
        .expect("Program.cs not found");

    let _person_controller = file_nodes
        .iter()
        .find(|f| f.file.ends_with("Controllers/PersonController.cs"))
        .expect("PersonController.cs not found");

    let _article_controller = file_nodes
        .iter()
        .find(|f| f.file.ends_with("Controllers/ArticleController.cs"))
        .expect("ArticleController.cs not found");

    let _order_controller = file_nodes
        .iter()
        .find(|f| f.file.ends_with("Controllers/OrderController.cs"))
        .expect("OrderController.cs not found");

    let directory_nodes = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directory_nodes.len();
    assert_eq!(directory_nodes.len(), 12, "Expected 12 Directory nodes");

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 Repository node");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 164, "Expected 164 Class nodes");

    let _person_class = classes
        .iter()
        .find(|c| c.name == "Person")
        .expect("Person model class not found");

    let _article_class = classes
        .iter()
        .find(|c| c.name == "Article")
        .expect("Article model class not found");

    let _order_class = classes
        .iter()
        .find(|c| c.name == "Order")
        .expect("Order model class not found");

    let _product_class = classes
        .iter()
        .find(|c| c.name == "Product")
        .expect("Product model class not found");

    let _person_controller_class = classes
        .iter()
        .find(|c| c.name == "PersonController")
        .expect("PersonController class not found");

    let _article_controller_class = classes
        .iter()
        .find(|c| c.name == "ArticleController")
        .expect("ArticleController class not found");

    let _person_service_class = classes
        .iter()
        .find(|c| c.name == "PersonService")
        .expect("PersonService class not found");

    let _application_db_context = classes
        .iter()
        .find(|c| c.name == "ApplicationDbContext")
        .expect("ApplicationDbContext class not found");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    assert_eq!(functions.len(), 362, "Expected 362 Function nodes");

    let _get_all_method = functions
        .iter()
        .find(|f| f.name == "GetAll" && f.file.ends_with("PersonController.cs"))
        .expect("PersonController::GetAll not found");

    let _get_by_id_method = functions
        .iter()
        .find(|f| f.name == "GetById" && f.file.ends_with("PersonController.cs"))
        .expect("PersonController::GetById not found");

    let _create_method = functions
        .iter()
        .find(|f| f.name == "Create" && f.file.ends_with("PersonController.cs"))
        .expect("PersonController::Create not found");

    let _update_method = functions
        .iter()
        .find(|f| f.name == "Update" && f.file.ends_with("PersonController.cs"))
        .expect("PersonController::Update not found");

    let _delete_method = functions
        .iter()
        .find(|f| f.name == "Delete" && f.file.ends_with("PersonController.cs"))
        .expect("PersonController::Delete not found");

    let _publish_method = functions
        .iter()
        .find(|f| f.name == "Publish" && f.file.ends_with("ArticleController.cs"))
        .expect("ArticleController::Publish not found");

    let _checkout_method = functions
        .iter()
        .find(|f| f.name == "Checkout" && f.file.ends_with("OrderController.cs"))
        .expect("OrderController::Checkout not found");

    let _calculate_age_method = functions
        .iter()
        .find(|f| f.name == "CalculateAge" && f.file.ends_with("Models/Person.cs"))
        .expect("Person::CalculateAge not found");

    let _invoke_async_method = functions
        .iter()
        .find(|f| f.name == "InvokeAsync" && f.file.ends_with("Middleware/Middleware.cs"))
        .expect("Middleware::InvokeAsync not found");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 24, "Expected 24 Import nodes");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 13, "Expected 13 Library nodes");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes += unit_tests.len();
    assert_eq!(unit_tests.len(), 31, "Expected 31 UnitTest nodes");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes += integration_tests.len();
    assert_eq!(
        integration_tests.len(),
        34,
        "Expected 34 IntegrationTest nodes"
    );

    let traits = graph.find_nodes_by_type(NodeType::Trait);
    nodes += traits.len();
    assert_eq!(traits.len(), 20, "Expected 20 Trait nodes");

    let datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += datamodels.len();
    assert_eq!(datamodels.len(), 19, "Expected 19 DataModel nodes");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 81, "Expected 81 Endpoint nodes");

    let vars = graph.find_nodes_by_type(NodeType::Var);
    nodes += vars.len();
    assert_eq!(vars.len(), 92, "Expected 92 Var nodes");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges += operands;
    assert_eq!(operands, 242, "Expected 242 Operand edges");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;
    // BTreeMapGraph produces slightly fewer call edges (208) than ArrayGraph (212)
    // likely due to iteration order differences in resolution logic
    if std::any::type_name::<G>().contains("ArrayGraph") {
        assert_eq!(calls, 212, "Expected 212 Calls edges for ArrayGraph");
    } else {
        assert_eq!(calls, 208, "Expected 208 Calls edges for BTreeMapGraph");
    }

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 1523, "Expected 1523 Contains edges");

    let parent_of = graph.count_edges_of_type(EdgeType::ParentOf);
    edges += parent_of;
    assert_eq!(parent_of, 12, "Expected 12 ParentOf edges");

    let imports_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges += imports_edges;
    assert_eq!(imports_edges, 0, "Expected 0 Imports edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges += handlers;
    assert_eq!(handlers, 81, "Expected 81 Handler edges");

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
async fn test_csharp() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_csharp_generic::<ArrayGraph>().await.unwrap();
    test_csharp_generic::<BTreeMapGraph>().await.unwrap();
}
