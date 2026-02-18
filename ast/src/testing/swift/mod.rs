use crate::lang::graphs::{ArrayGraph, BTreeMapGraph, EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::slice_body;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_swift_legacy_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/swift/LegacyApp",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "swift",
        "Language node name should be 'swift'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/swift/LegacyApp",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 25, "Expected 25 files");

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "Podfile");
    assert_eq!(pkg_files.len(), 1, "Expected 1 Podfile");
    let expected_results = r#"platform :ios, '15.0'
use_frameworks!
inhibit_all_warnings!

install! 'cocoapods'

target 'SphinxTestApp' do
    use_frameworks!
    pod 'Alamofire', '~> 5.10.2'
    pod 'SwiftyJSON'
    pod 'ObjectMapper'
end
"#
    .to_string();
    assert_eq!(
        pkg_files[0].name, "Podfile",
        "Package file name is incorrect"
    );
    assert_eq!(
        slice_body(
            &std::fs::read_to_string(&pkg_files[0].file).expect("Failed to read file"),
            pkg_files[0].start,
            pkg_files[0].end
        ),
        expected_results,
        "Podfile should contain correct content"
    );

    let info_plist = graph.find_nodes_by_name(NodeType::File, "Info.plist");
    assert_eq!(info_plist.len(), 1, "Expected 1 Info.plist");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 7, "Expected 7 imports");

    let ui_kit_import = imports
        .iter()
        .find(|i| {
            let code = std::fs::read_to_string(&i.file).expect("Failed to read file");
            let body = slice_body(&code, i.start, i.end);
            body.contains("UIKit")
        })
        .expect("UIKit import not found");
    let ui_kit_body = slice_body(
        &std::fs::read_to_string(&ui_kit_import.file).expect("Failed to read file"),
        ui_kit_import.start,
        ui_kit_import.end,
    );
    assert!(
        ui_kit_body.contains("import UIKit"),
        "Should import UIKit framework"
    );

    let core_data_import = imports
        .iter()
        .find(|i| {
            let code = std::fs::read_to_string(&i.file).expect("Failed to read file");
            let body = slice_body(&code, i.start, i.end);
            body.contains("CoreData")
        })
        .expect("CoreData import not found");
    let core_data_body = slice_body(
        &std::fs::read_to_string(&core_data_import.file).expect("Failed to read file"),
        core_data_import.start,
        core_data_import.end,
    );
    assert!(
        core_data_body.contains("import CoreData"),
        "Should import CoreData framework"
    );

    let foundation_import = imports
        .iter()
        .find(|i| {
            let code = std::fs::read_to_string(&i.file).expect("Failed to read file");
            let body = slice_body(&code, i.start, i.end);
            body.contains("Foundation")
        })
        .expect("Foundation import not found");
    let foundation_body = slice_body(
        &std::fs::read_to_string(&foundation_import.file).expect("Failed to read file"),
        foundation_import.start,
        foundation_import.end,
    );
    assert!(
        foundation_body.contains("import Foundation"),
        "Should import Foundation framework"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 9, "Expected 9 classes");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_classes[0].name, "API",
        "First class name should be 'API'"
    );

    let app_delegate = classes
        .iter()
        .find(|c| c.name == "AppDelegate")
        .expect("AppDelegate class not found");
    assert!(
        app_delegate.file.contains("AppDelegate.swift"),
        "AppDelegate should be in AppDelegate.swift file"
    );

    let view_controller = classes
        .iter()
        .find(|c| c.name == "ViewController")
        .expect("ViewController class not found");
    assert!(
        view_controller.file.contains("ViewController.swift"),
        "ViewController should be in ViewController.swift file"
    );

    let person_class = classes
        .iter()
        .find(|c| c.name == "Person" && c.file.contains("Person+CoreDataClass.swift"))
        .expect("Person CoreData class not found");
    assert!(
        person_class.file.contains("CoreData"),
        "Person class should be in CoreData directory"
    );

    let api_class = classes
        .iter()
        .find(|c| c.name == "API")
        .expect("API class not found");
    assert!(
        api_class.file.contains("API.swift"),
        "API class should be in API.swift file"
    );

    let scene_delegate = classes
        .iter()
        .find(|c| c.name == "SceneDelegate")
        .expect("SceneDelegate class not found");
    assert!(
        scene_delegate.file.contains("SceneDelegate.swift"),
        "SceneDelegate should be in SceneDelegate.swift file"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 26, "Expected 26 functions");

    let mut sorted_functions = functions.clone();
    sorted_functions.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_functions[0].name, "application",
        "First function name should be 'application'"
    );

    let application_launch = functions
        .iter()
        .find(|f| f.name == "application" && f.file.contains("AppDelegate.swift"))
        .expect("application function not found");
    assert!(
        application_launch.file.contains("AppDelegate.swift"),
        "application function should be in AppDelegate.swift"
    );

    let save_context = functions
        .iter()
        .find(|f| f.name == "saveContext")
        .expect("saveContext function not found");
    let save_context_body = slice_body(
        &std::fs::read_to_string(&save_context.file).expect("Failed to read file"),
        save_context.start,
        save_context.end,
    );
    assert!(
        save_context_body.contains("context"),
        "saveContext should use context"
    );

    let table_view_cell = functions
        .iter()
        .find(|f| f.name == "tableView" && f.file.contains("ViewController.swift"))
        .expect("tableView function not found");
    assert!(
        table_view_cell.file.contains("ViewController.swift"),
        "tableView function should be in ViewController.swift"
    );

    let fetch_all_objects = functions
        .iter()
        .find(|f| f.name == "fetchAllObjects")
        .expect("fetchAllObjects function not found");
    assert!(
        fetch_all_objects.file.contains("CoreData"),
        "fetchAllObjects should be in CoreData directory"
    );

    let get_people = functions
        .iter()
        .find(|f| f.name == "getPeopleList")
        .expect("getPeopleList function not found");
    assert!(
        get_people.file.contains("API.swift"),
        "getPeopleList should be in API.swift file"
    );
    assert_eq!(
        get_people.docs,
        Some("Get the list of people.".to_string()),
        "getPeopleList should have documentation"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data model");

    let person_data_model = &data_models[0];
    assert_eq!(
        person_data_model.name, "Person",
        "Data model name should be Person"
    );
    assert!(
        person_data_model
            .file
            .contains("Person+CoreDataClass.swift"),
        "Data model should be in Person+CoreDataClass.swift file"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let mut sorted_requests = requests.clone();
    sorted_requests.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_requests[0].name, "/people",
        "First request name should be '/people'"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 2, "Expected 2 variables");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 19, "Expected 19 directories");

    let sphinx_test_app_dir = directories
        .iter()
        .find(|d| d.name == "SphinxTestApp")
        .expect("SphinxTestApp directory not found");

    let core_data_dir = directories
        .iter()
        .find(|d| d.name == "CoreData")
        .expect("CoreData directory not found");

    let dir_relationship = graph.has_edge(
        &Node::new(NodeType::Directory, sphinx_test_app_dir.clone()),
        &Node::new(NodeType::Directory, core_data_dir.clone()),
        EdgeType::Contains,
    );
    assert!(
        dir_relationship,
        "Expected Contains edge between SphinxTestApp and CoreData directories"
    );

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls;
    assert_eq!(calls, 2, "Expected 2 call edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(contains, 90, "Expected 90 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 Of edges");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operands;
    assert_eq!(operands, 22, "Expected 22 operand edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    assert_eq!(handlers, 0, "Expected 0 handler edges");

    let operand_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(
        operand_edges.len(),
        22,
        "Expected at least 22 operand edges"
    );

    let api_operand = operand_edges
        .iter()
        .any(|(src, dst)| src.name == "API" && dst.name == "createRequest");
    assert!(
        api_operand,
        "Expected API -> Operand -> createRequest method edge"
    );

    let view_controller_operand = operand_edges
        .iter()
        .any(|(src, dst)| src.name == "ViewController" && dst.name == "viewDidLoad");
    assert!(
        view_controller_operand,
        "Expected ViewController -> Operand -> viewDidLoad method edge"
    );

    let scene_delegate_operand = operand_edges
        .iter()
        .any(|(src, dst)| src.name == "SceneDelegate" && dst.name == "scene");
    assert!(
        scene_delegate_operand,
        "Expected SceneDelegate -> Operand -> scene method edge"
    );

    let call_edges =
        graph.find_nodes_with_edge_type(NodeType::Function, NodeType::Request, EdgeType::Calls);
    assert_eq!(
        call_edges.len(),
        2,
        "Expected 2 function to request call edges"
    );

    let get_people_call = call_edges
        .iter()
        .any(|(src, dst)| src.name == "getPeopleList" && dst.name == "/people");
    assert!(
        get_people_call,
        "Expected getPeopleList -> Calls -> people request edge"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(nodes as usize, nodes_count, "Node count mismatch");
    assert_eq!(edges as usize, edges_count, "Edge count mismatch");

    Ok(())
}
pub async fn test_swift_modern_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/swift/ModernApp",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "swift",
        "Language node name should be 'swift'"
    );

    let structs = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += structs.len();

    let profile_service = structs
        .iter()
        .find(|s| s.name == "ProfileService")
        .expect("ProfileService actor not found");
    assert!(
        profile_service.file.contains("ProfileService.swift"),
        "ProfileService should be in ProfileService.swift"
    );

    let profile_view = structs
        .iter()
        .find(|s| s.name == "ProfileView")
        .expect("ProfileView struct not found");
    assert!(
        profile_view.file.contains("ProfileView.swift"),
        "ProfileView should be in ProfileView.swift"
    );

    let _api_client = structs
        .iter()
        .find(|s| s.name == "APIClient")
        .expect("APIClient struct not found");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();

    let fetch_profile = functions
        .iter()
        .find(|f| f.name == "fetchProfile")
        .expect("fetchProfile function not found");
    let fetch_profile_body = slice_body(
        &std::fs::read_to_string(&fetch_profile.file).expect("Failed to read file"),
        fetch_profile.start,
        fetch_profile.end,
    );
    assert!(
        fetch_profile_body.contains("async"),
        "fetchProfile should be async"
    );
    assert!(
        fetch_profile_body.contains("throws"),
        "fetchProfile should throw"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 2, "Expected 2 unit tests");
    let _test_fetch = unit_tests
        .iter()
        .find(|t| t.name == "testFetchProfile")
        .expect("testFetchProfile not found");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    let _package_file = files
        .iter()
        .find(|f| f.name == "Package.swift")
        .expect("Package.swift not found");

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository");

    let file_imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += file_imports.len();
    assert_eq!(file_imports.len(), 12, "Expected 12 imports");

    let vars = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += vars.len();
    assert_eq!(vars.len(), 1, "Expected 1 variable");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 10, "Expected 10 directories");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges;
    assert_eq!(contains_edges, 60, "Expected 60 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 Of edge");

    let operand_edges = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges;
    assert_eq!(operand_edges, 7, "Expected 7 operand edges");

    let call_edges = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += call_edges;
    assert_eq!(call_edges, 0, "Expected 0 call edges");

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(nodes as usize, nodes_count, "Node count mismatch");
    assert_eq!(edges as usize, edges_count, "Edge count mismatch");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_swift() {
    test_swift_legacy_generic::<ArrayGraph>().await.unwrap();
    test_swift_legacy_generic::<BTreeMapGraph>().await.unwrap();
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_swift_legacy_generic::<Neo4jGraph>().await.unwrap();
    }

    test_swift_modern_generic::<ArrayGraph>().await.unwrap();
    test_swift_modern_generic::<BTreeMapGraph>().await.unwrap();
}
