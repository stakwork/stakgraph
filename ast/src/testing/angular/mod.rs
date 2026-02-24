use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_angular_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/angular",
        Lang::from_str("angular").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let mut nodes = 0;
    let mut edges = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "angular",
        "Language node name should be 'angular'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/angular",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes += files.len();
    assert_eq!(files.len(), 29, "Expected 29 files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directories.len();
    assert_eq!(directories.len(), 7, "Expected 7 directory");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 26, "Expected 26 library node");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;
    assert_eq!(calls, 15, "Expected 15 call edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 106, "Expected 106 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 of edges");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 14, "Expected 14 imports");

    let main_import_body = format!(
        r#"import {{ bootstrapApplication }} from '@angular/platform-browser';
import {{ appConfig }} from './app/app.config';
import {{ AppComponent }} from './app/app.component';"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/angular/src/main.ts")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 5, "Expected 5 classes");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert!(
        classes.iter().any(|c| c.name == "AppComponent"
            && c.file == "src/testing/angular/src/app/app.component.ts"),
        "Expected AppComponent class not found"
    );

    assert!(
        classes.iter().any(|c| c.name == "PeopleService"
            && c.file == "src/testing/angular/src/app/people.service.ts"),
        "Expected PeopleService class not found"
    );

    assert!(
        classes.iter().any(|c| c.name == "PeopleListComponent"
            && c.file == "src/testing/angular/src/app/people-list/people-list.component.ts"),
        "Expected PeopleListComponent class not found"
    );

    assert!(
        classes.iter().any(|c| c.name == "AddPersonComponent"
            && c.file == "src/testing/angular/src/app/add-person/add-person.component.ts"),
        "Expected AddPersonComponent class not found"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();

    let log_method_fn = functions
        .iter()
        .find(|f| {
            f.name == "LogMethod" && f.file == "src/testing/angular/src/app/people.service.ts"
        })
        .expect("LogMethod function not found in people.service.ts");

    assert!(
        log_method_fn.meta.contains_key("interface"),
        "LogMethod function should have interface metadata"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data model");
    assert_eq!(
        data_models[0].name, "Person",
        "Data model name should be 'Person'"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes += requests.len();
    assert_eq!(requests.len(), 10, "Expected 10 requests");

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes += unit_tests.len();
    assert_eq!(
        unit_tests.len(),4,
        "Expected  4 unit tests from Angular spec files"
    );
    assert!(
        unit_tests.iter().all(|t| t.file.contains(".spec.")),
        "Expected Angular unit tests to come from spec files"
    );

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes += integration_tests.len();
    assert_eq!(
        integration_tests.len(),
        0,
        "Expected 0 integration tests for current Angular fixture"
    );

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes += e2e_tests.len();
    assert_eq!(
        e2e_tests.len(),
        0,
        "Expected 0 e2e tests for current Angular fixture"
    );

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges += imported_edges;
    assert_eq!(imported_edges, 12, "Expected 12 import edges");

    let person_data_model = graph
        .find_nodes_by_name(NodeType::DataModel, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/models/person.model.ts")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("Person DataModel not found in person.model.ts");

    let person_model_file = graph
        .find_nodes_by_name(NodeType::File, "person.model.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/models/person.model.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("person.model.ts file node not found");

    let app_component_class = graph
        .find_nodes_by_name(NodeType::Class, "AppComponent")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.component.ts")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("AppComponent class not found");

    let people_service_class = graph
        .find_nodes_by_name(NodeType::Class, "PeopleService")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people.service.ts")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("PeopleService class not found");

    let people_list_component_class = graph
        .find_nodes_by_name(NodeType::Class, "PeopleListComponent")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people-list/people-list.component.ts")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("PeopleListComponent class not found");

    let add_person_component_class = graph
        .find_nodes_by_name(NodeType::Class, "AddPersonComponent")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/add-person/add-person.component.ts")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("AddPersonComponent class not found");

    assert!(
        graph.has_edge(&person_model_file, &person_data_model, EdgeType::Contains),
        "Expected 'person.model.ts' file to contain 'Person' DataModel"
    );

    let app_component_file = graph
        .find_nodes_by_name(NodeType::File, "app.component.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.component.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("app.component.ts file not found");

    let people_service_file = graph
        .find_nodes_by_name(NodeType::File, "people.service.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people.service.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("people.service.ts file not found");

    let people_list_component_file = graph
        .find_nodes_by_name(NodeType::File, "people-list.component.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people-list/people-list.component.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("people-list.component.ts file not found");

    let add_person_component_file = graph
        .find_nodes_by_name(NodeType::File, "add-person.component.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/add-person/add-person.component.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("add-person.component.ts file not found");

    assert!(
        graph.has_edge(
            &app_component_file,
            &app_component_class,
            EdgeType::Contains
        ),
        "Expected app.component.ts to contain AppComponent class"
    );

    assert!(
        graph.has_edge(
            &people_service_file,
            &people_service_class,
            EdgeType::Contains
        ),
        "Expected people.service.ts to contain PeopleService class"
    );

    assert!(
        graph.has_edge(
            &people_list_component_file,
            &people_list_component_class,
            EdgeType::Contains
        ),
        "Expected people-list.component.ts to contain PeopleListComponent class"
    );

    assert!(
        graph.has_edge(
            &add_person_component_file,
            &add_person_component_class,
            EdgeType::Contains
        ),
        "Expected add-person.component.ts to contain AddPersonComponent class"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 10, "Expected 10 functions");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 4, "Expected 4 variables");

    let constructor_fn = graph
        .find_nodes_by_name(NodeType::Function, "constructor")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people.service.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("PeopleService constructor not found");

    let add_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "addPerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people.service.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("addPerson method not found in PeopleService");

    let delete_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "deletePerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people.service.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("deletePerson method not found in PeopleService");

    let ng_on_init_fn = graph
        .find_nodes_by_name(NodeType::Function, "ngOnInit")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people-list/people-list.component.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("ngOnInit method not found in PeopleListComponent");

    let delete_person_component_fn = graph
        .find_nodes_by_name(NodeType::Function, "deletePerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people-list/people-list.component.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("deletePerson method not found in PeopleListComponent");

    let add_person_component_fn = graph
        .find_nodes_by_name(NodeType::Function, "addPerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/add-person/add-person.component.ts")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("addPerson method not found in AddPersonComponent");

    assert!(
        graph.has_edge(&people_service_file, &constructor_fn, EdgeType::Contains),
        "Expected people.service.ts to contain constructor"
    );
    assert!(
        graph.has_edge(&people_service_file, &add_person_fn, EdgeType::Contains),
        "Expected people.service.ts to contain addPerson method"
    );

    assert!(
        graph.has_edge(&people_service_file, &delete_person_fn, EdgeType::Contains),
        "Expected people.service.ts to contain deletePerson method"
    );

    assert!(
        graph.has_edge(
            &people_list_component_file,
            &ng_on_init_fn,
            EdgeType::Contains
        ),
        "Expected people-list.component.ts to contain ngOnInit method"
    );
    assert!(
        graph.has_edge(
            &people_list_component_file,
            &delete_person_component_fn,
            EdgeType::Contains
        ),
        "Expected people-list.component.ts to contain deletePerson method"
    );

    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    edges += renders_edges_count;
    assert_eq!(renders_edges_count, 7, "Expected 7 RENDERS edge");

    assert!(
        graph.has_edge(
            &add_person_component_file,
            &add_person_component_fn,
            EdgeType::Contains
        ),
        "Expected add-person.component.ts to contain addPerson method"
    );

    let add_person_service_request = graph
        .find_nodes_by_name(NodeType::Request, "this.peopleService.addPerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/add-person/add-person.component.ts")
        .map(|n| Node::new(NodeType::Request, n))
        .expect("addPerson service request not found in AddPersonComponent");

    let delete_person_service_request = graph
        .find_nodes_by_name(NodeType::Request, "this.peopleService.deletePerson")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/people-list/people-list.component.ts")
        .map(|n| Node::new(NodeType::Request, n))
        .expect("deletePerson service request not found in PeopleListComponent");

    assert!(
        graph.has_edge(
            &delete_person_component_fn,
            &delete_person_service_request,
            EdgeType::Calls
        ),
        "Expected PeopleListComponent deletePerson to call deletePerson service request"
    );

    assert!(
        graph.has_edge(
            &add_person_component_fn,
            &add_person_service_request,
            EdgeType::Calls
        ),
        "Expected AddPersonComponent addPerson to call addPerson service request"
    );

    let app_routes_file = graph
        .find_nodes_by_name(NodeType::File, "app.routes.ts")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.routes.ts")
        .map(|n| Node::new(NodeType::File, n))
        .expect("app.routes.ts file not found");

    let routes_var = graph
        .find_nodes_by_name(NodeType::Var, "routes")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.routes.ts")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("routes variable not found in app.routes.ts");

    assert!(
        graph.has_edge(&app_routes_file, &routes_var, EdgeType::Contains),
        "Expected app.routes.ts to contain routes variable"
    );

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes += pages.len();
    assert_eq!(pages.len(), 11, "Expected 11 pages");

    let index_page_nodes = graph.find_nodes_by_file_ends_with(NodeType::Page, "src/index.html");
    assert_eq!(
        index_page_nodes.len(),
        1,
        "Expected to find the index.html page"
    );

    let app_component_page_nodes =
        graph.find_nodes_by_file_ends_with(NodeType::Page, "src/app/app.component.html");
    assert_eq!(
        app_component_page_nodes.len(),
        1,
        "Expected to find the app.component.html page"
    );

    let people_list_page_nodes = graph.find_nodes_by_file_ends_with(
        NodeType::Page,
        "src/app/people-list/people-list.component.html",
    );

    assert_eq!(
        people_list_page_nodes.len(),
        1,
        "Expected to find the people-list.component.html page"
    );
    let app_component_page = app_component_page_nodes.first().unwrap();

    let add_person_page_nodes = graph.find_nodes_by_file_ends_with(
        NodeType::Page,
        "src/app/people-list/people-list.component.html",
    );
    assert_eq!(
        add_person_page_nodes.len(),
        1,
        "Expected to find the add-person.component.html page"
    );
    let people_list_page = people_list_page_nodes.first().unwrap();

    let app_component_node = Node::new(NodeType::Page, app_component_page.clone());
    let people_list_node = Node::new(NodeType::Page, people_list_page.clone());

    let has_render_edge = graph.has_edge(&app_component_node, &people_list_node, EdgeType::Renders);
    assert!(
        has_render_edge,
        "Expected app.component.html to render people-list.component.html"
    );

    let vars = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(vars.len(), 4, "Expected 4 variables");

    let app_config_var = graph
        .find_nodes_by_name(NodeType::Var, "appConfig")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.config.ts")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("appConfig variable not found");

    let server_config_var = graph
        .find_nodes_by_name(NodeType::Var, "serverConfig")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/app/app.config.server.ts")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("serverConfig variable not found");

    let common_engine_var = graph
        .find_nodes_by_name(NodeType::Var, "commonEngine")
        .into_iter()
        .find(|n| n.file == "src/testing/angular/src/server.ts")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("commonEngine variable not found");

    // Verify variable containment
    assert!(
        graph.has_edge(
            &graph
                .find_nodes_by_name(NodeType::File, "app.config.ts")
                .into_iter()
                .find(|n| n.file == "src/testing/angular/src/app/app.config.ts")
                .map(|n| Node::new(NodeType::File, n))
                .expect("app.config.ts file not found"),
            &app_config_var,
            EdgeType::Contains
        ),
        "Expected app.config.ts to contain appConfig variable"
    );

    assert!(
        graph.has_edge(
            &graph
                .find_nodes_by_name(NodeType::File, "app.config.server.ts")
                .into_iter()
                .find(|n| n.file == "src/testing/angular/src/app/app.config.server.ts")
                .map(|n| Node::new(NodeType::File, n))
                .expect("app.config.server.ts file not found"),
            &server_config_var,
            EdgeType::Contains
        ),
        "Expected app.config.server.ts to contain serverConfig variable"
    );

    assert!(
        graph.has_edge(
            &graph
                .find_nodes_by_name(NodeType::File, "server.ts")
                .into_iter()
                .find(|n| n.file == "src/testing/angular/src/server.ts")
                .map(|n| Node::new(NodeType::File, n))
                .expect("server.ts file not found"),
            &common_engine_var,
            EdgeType::Contains
        ),
        "Expected server.ts to contain commonEngine variable"
    );

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(
        num_nodes, nodes as u32,
        "Expected {} nodes, found {}",
        nodes, num_nodes
    );
    assert_eq!(
        num_edges, edges as u32,
        "Expected {} edges, found {}",
        edges, num_edges
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_angular() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_angular_generic::<ArrayGraph>().await.unwrap();
        test_angular_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_angular_generic::<Neo4jGraph>().await.unwrap();
    }
}
