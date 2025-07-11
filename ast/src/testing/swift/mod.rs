use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;

pub async fn test_swift_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let repo = Repo::new(
        "src/testing/swift",
        Lang::from_str("swift").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 91, "Expected 91 nodes");
    assert_eq!(num_edges, 117, "Expected 117 edges");

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "swift",
        "Language node name should be 'swift'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/swift",
        "Language node file path is incorrect"
    );

    let files = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(files.len(), 25, "Expected 25 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    assert_eq!(imports.len(), 7, "Expected 7 imports");

    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 7, "Expected 7 classes");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 26, "Expected 26 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 2, "Expected 2 variables");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls, 2, "Expected 2 call edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(contains, 89, "Expected 89 contains edges");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    assert_eq!(operands, 26, "Expected 26 operand edges");

    let api_file = graph
        .find_nodes_by_name(NodeType::File, "API.swift")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::File, n))
        .expect("API.swift file not found");

    let view_controller_file = graph
        .find_nodes_by_name(NodeType::File, "ViewController.swift")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/ViewController.swift")
        .map(|n| Node::new(NodeType::File, n))
        .expect("ViewController.swift file not found");

    let app_delegate_file = graph
        .find_nodes_by_name(NodeType::File, "AppDelegate.swift")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/AppDelegate.swift")
        .map(|n| Node::new(NodeType::File, n))
        .expect("AppDelegate.swift file not found");

    let person_core_data_file = graph
        .find_nodes_by_name(NodeType::File, "Person+CoreDataClass.swift")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/CoreData/Person+CoreDataClass.swift")
        .map(|n| Node::new(NodeType::File, n))
        .expect("Person+CoreDataClass.swift file not found");

    let api_class = graph
        .find_nodes_by_name(NodeType::Class, "API")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("API class not found");

    let view_controller_class = graph
        .find_nodes_by_name(NodeType::Class, "ViewController")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/ViewController.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("ViewController class not found");

    let app_delegate_class = graph
        .find_nodes_by_name(NodeType::Class, "AppDelegate")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/AppDelegate.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("AppDelegate class not found");

    let person_class = graph
        .find_nodes_by_name(NodeType::Class, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/CoreData/Person+CoreDataClass.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("Person class not found");

    let scene_delegate_class = graph
        .find_nodes_by_name(NodeType::Class, "SceneDelegate")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/SceneDelegate.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("SceneDelegate class not found");

    assert!(
        graph.has_edge(&api_file, &api_class, EdgeType::Contains),
        "Expected API.swift to contain API class"
    );

    assert!(
        graph.has_edge(
            &view_controller_file,
            &view_controller_class,
            EdgeType::Contains
        ),
        "Expected ViewController.swift to contain ViewController class"
    );

    assert!(
        graph.has_edge(&app_delegate_file, &app_delegate_class, EdgeType::Contains),
        "Expected AppDelegate.swift to contain AppDelegate class"
    );

    assert!(
        graph.has_edge(&person_core_data_file, &person_class, EdgeType::Contains),
        "Expected Person+CoreDataClass.swift to contain Person class"
    );

    let person_data_model = graph
        .find_nodes_by_name(NodeType::DataModel, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/CoreData/Person+CoreDataClass.swift")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("Person DataModel not found");

    assert!(
        graph.has_edge(
            &person_core_data_file,
            &person_data_model,
            EdgeType::Contains
        ),
        "Expected Person+CoreDataClass.swift to contain Person DataModel"
    );

    let get_people_list_fn = graph
        .find_nodes_by_name(NodeType::Function, "getPeopleList")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("getPeopleList function not found");

    let update_people_profile_fn = graph
        .find_nodes_by_name(NodeType::Function, "updatePeopleProfileWith")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("updatePeopleProfileWith function not found");

    let view_did_load_fn = graph
        .find_nodes_by_name(NodeType::Function, "viewDidLoad")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/ViewController.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("viewDidLoad function not found");

    let get_people_and_save_fn = graph
        .find_nodes_by_name(NodeType::Function, "getPeopleAndSave")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/ViewController.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("getPeopleAndSave function not found");

    let application_fn = graph
        .find_nodes_by_name(NodeType::Function, "application")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/AppDelegate.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("application function not found");

    assert!(
        graph.has_edge(&api_file, &get_people_list_fn, EdgeType::Contains),
        "Expected API.swift to contain getPeopleList function"
    );

    assert!(
        graph.has_edge(&api_file, &update_people_profile_fn, EdgeType::Contains),
        "Expected API.swift to contain updatePeopleProfileWith function"
    );

    assert!(
        graph.has_edge(&view_controller_file, &view_did_load_fn, EdgeType::Contains),
        "Expected ViewController.swift to contain viewDidLoad function"
    );

    assert!(
        graph.has_edge(
            &view_controller_file,
            &get_people_and_save_fn,
            EdgeType::Contains
        ),
        "Expected ViewController.swift to contain getPeopleAndSave function"
    );

    assert!(
        graph.has_edge(&app_delegate_file, &application_fn, EdgeType::Contains),
        "Expected AppDelegate.swift to contain application function"
    );

    assert!(
        graph.has_edge(&api_class, &get_people_list_fn, EdgeType::Operand),
        "Expected API class to have getPeopleList method"
    );

    assert!(
        graph.has_edge(&api_class, &update_people_profile_fn, EdgeType::Operand),
        "Expected API class to have updatePeopleProfileWith method"
    );

    assert!(
        graph.has_edge(&view_controller_class, &view_did_load_fn, EdgeType::Operand),
        "Expected ViewController class to have viewDidLoad method"
    );

    assert!(
        graph.has_edge(
            &view_controller_class,
            &get_people_and_save_fn,
            EdgeType::Operand
        ),
        "Expected ViewController class to have getPeopleAndSave method"
    );

    let people_request = graph
        .find_nodes_by_name(NodeType::Request, "/people")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::Request, n))
        .expect("/people request not found");

    let person_request = graph
        .find_nodes_by_name(NodeType::Request, "/person")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/API.swift")
        .map(|n| Node::new(NodeType::Request, n))
        .expect("/person request not found");

    assert!(
        graph.has_edge(&get_people_list_fn, &people_request, EdgeType::Calls),
        "Expected getPeopleList to call /people endpoint"
    );

    assert!(
        graph.has_edge(&update_people_profile_fn, &person_request, EdgeType::Calls),
        "Expected updatePeopleProfileWith to call /person endpoint"
    );

    assert!(
        graph.has_edge(
            &get_people_and_save_fn,
            &person_data_model,
            EdgeType::Contains
        ),
        "Expected getPeopleAndSave to contain Person DataModel"
    );

    let name_var = graph
        .find_nodes_by_name(NodeType::Var, "Name")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/AppDelegate.swift")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("Name variable not found");

    let version_var = graph
        .find_nodes_by_name(NodeType::Var, "Version")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/AppDelegate.swift")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("Version variable not found");

    assert!(
        graph.has_edge(&app_delegate_file, &name_var, EdgeType::Contains),
        "Expected AppDelegate.swift to contain Name variable"
    );

    assert!(
        graph.has_edge(&app_delegate_file, &version_var, EdgeType::Contains),
        "Expected AppDelegate.swift to contain Version variable"
    );

    let person_table_view_cell_class = graph
        .find_nodes_by_name(NodeType::Class, "PersonTableViewCell")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/PersonTableViewCell.swift")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("PersonTableViewCell class not found");

    let awake_from_nib_fn = graph
        .find_nodes_by_name(NodeType::Function, "awakeFromNib")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/PersonTableViewCell.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("awakeFromNib function not found");

    let set_selected_fn = graph
        .find_nodes_by_name(NodeType::Function, "setSelected")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/PersonTableViewCell.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("setSelected function not found");

    assert!(
        graph.has_edge(
            &person_table_view_cell_class,
            &awake_from_nib_fn,
            EdgeType::Operand
        ),
        "Expected PersonTableViewCell to have awakeFromNib method"
    );

    assert!(
        graph.has_edge(
            &person_table_view_cell_class,
            &set_selected_fn,
            EdgeType::Operand
        ),
        "Expected PersonTableViewCell to have setSelected method"
    );

    let scene_fn = graph
        .find_nodes_by_name(NodeType::Function, "scene")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/SceneDelegate.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("scene function not found");

    let scene_did_disconnect_fn = graph
        .find_nodes_by_name(NodeType::Function, "sceneDidDisconnect")
        .into_iter()
        .find(|n| n.file == "src/testing/swift/SphinxTestApp/SceneDelegate.swift")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("sceneDidDisconnect function not found");

    assert!(
        graph.has_edge(&scene_delegate_class, &scene_fn, EdgeType::Operand),
        "Expected SceneDelegate to have scene method"
    );

    assert!(
        graph.has_edge(
            &scene_delegate_class,
            &scene_did_disconnect_fn,
            EdgeType::Operand
        ),
        "Expected SceneDelegate to have sceneDidDisconnect method"
    );

    let table_view_functions = graph
        .find_nodes_by_name(NodeType::Function, "tableView")
        .iter()
        .filter(|n| n.file == "src/testing/swift/SphinxTestApp/ViewController.swift")
        .count();
    assert_eq!(
        table_view_functions, 3,
        "Expected 3 tableView functions in ViewController"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_swift() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_swift_generic::<ArrayGraph>().await.unwrap();
    test_swift_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_swift_generic::<Neo4jGraph>().await.unwrap();
    }
}
