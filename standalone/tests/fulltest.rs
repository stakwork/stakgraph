#[cfg(feature = "fulltest")]
use ast::lang::graphs::{ArrayGraph, BTreeMapGraph};

#[cfg(feature = "fulltest")]
use ast::lang::{EdgeType, Graph, Node, NodeType};
#[cfg(feature = "fulltest")]
use ast::repo::Repo;
#[cfg(feature = "fulltest")]
use ast::utils::get_use_lsp;
#[cfg(feature = "fulltest")]
use test_log::test;
#[cfg(feature = "fulltest")]
use tracing::info;

#[cfg(feature = "fulltest")]
const REPO_URL: &str = "https://github.com/fayekelmith/demorepo.git";
#[cfg(feature = "fulltest")]
const COMMIT: &str = "778b5202fca04a2cd5daed377c0063e9af52b24c";
#[cfg(feature = "fulltest")]
async fn fulltest_generic<G: Graph>(graph: &G, use_lsp: bool) {
    info!(
        "Running fulltest for {} with LSP={}",
        std::any::type_name::<G>(),
        use_lsp
    );

    let (num_nodes, num_edges) = graph.get_graph_size();
    graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;
    /*****
     *
     ** NODES TESTS **
     *
     *****/

    /* REPOSITORY */
    if use_lsp {
        if graph_type_name.contains("ArrayGraph") {
            assert_eq!(num_nodes, 114, "Expected 157 nodes for ArrayGraph with LSP");
            assert_eq!(num_edges, 159, "Expected 159 edges for ArrayGraph with LSP");
        } else if graph_type_name.contains("BTreeMapGraph")
            || graph_type_name.contains("Neo4jGraph")
        {
            assert_eq!(
                num_nodes, 102,
                "Expected 102 nodes for BTreeMapGraph with LSP"
            );
            assert_eq!(
                num_edges, 148,
                "Expected 148 edges for BTreeMapGraph with LSP"
            );
        }
    } else {
        if graph_type_name.contains("ArrayGraph") {
            assert_eq!(
                num_nodes, 114,
                "Expected 114 nodes for ArrayGraph without LSP"
            );
            assert_eq!(
                num_edges, 155,
                "Expected 155 edges for ArrayGraph without LSP"
            );
        } else if graph_type_name.contains("BTreeMapGraph")
            || graph_type_name.contains("Neo4jGraph")
        {
            assert_eq!(
                num_nodes, 102,
                "Expected 102 nodes for BTreeMapGraph without LSP"
            );
            assert_eq!(
                num_edges, 144,
                "Expected 144 edges for BTreeMapGraph without LSP"
            );
        }
    }

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repositories.len();

    let repo_node = &repositories[0];
    assert_eq!(
        repo_node.name, "fayekelmith/demorepo",
        "Repository name is incorrect"
    );
    assert_eq!(
        repo_node.file, "fayekelmith/demorepo",
        "Repository file is incorrect"
    );

    /* LANGUAGE TESTS */

    let languages = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += languages.len();
    assert_eq!(languages.len(), 2, "Expected 2 language nodes");

    let go_lang = languages
        .iter()
        .find(|l| l.name == "go")
        .expect("Go language node not found");
    assert_eq!(go_lang.name, "go", "Go language name is incorrect");

    let react_lang = languages
        .iter()
        .find(|l| l.name == "react")
        .expect("React language node not found");
    assert_eq!(react_lang.name, "react", "React language name is incorrect");

    /* FILE SYSTEM TESTS */

    /* DIRECTORY */

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 4, "Expected 4 directories");

    let frontend_dir = directories
        .iter()
        .find(|d| d.name == "frontend")
        .expect("Frontend directory not found");
    assert!(
        frontend_dir.file.contains("frontend"),
        "Frontend directory path is incorrect"
    );

    let public_dir = directories
        .iter()
        .find(|d| d.name == "public")
        .expect("Public directory not found");
    assert!(
        public_dir.file.contains("frontend/public"),
        "Public directory path is incorrect"
    );

    let src_dir = directories
        .iter()
        .find(|d| d.name == "src")
        .expect("Src directory not found");
    assert!(
        src_dir.file.contains("frontend/src"),
        "Src directory path is incorrect"
    );

    let components_dir = directories
        .iter()
        .find(|d| d.name == "components")
        .expect("Components directory not found");
    assert!(
        components_dir.file.contains("frontend/src/components"),
        "Components directory path is incorrect"
    );

    /* FILES */

    /* GO FILES */
    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 22, "Expected 22 files");

    let go_files: Vec<_> = files.iter().filter(|f| f.name.ends_with(".go")).collect();
    assert_eq!(go_files.len(), 5, "Expected 5 Go files");

    let main_go = files
        .iter()
        .find(|f| f.name == "main.go")
        .expect("main.go not found");
    assert!(
        main_go.file.contains("main.go"),
        "main.go path is incorrect"
    );

    let db_go = files
        .iter()
        .find(|f| f.name == "db.go")
        .expect("db.go not found");
    assert!(db_go.file.contains("db.go"), "db.go path is incorrect");

    let routes_go = files
        .iter()
        .find(|f| f.name == "routes.go")
        .expect("routes.go not found");
    assert!(
        routes_go.file.contains("routes.go"),
        "routes.go path is incorrect"
    );

    let alpha_go = files
        .iter()
        .find(|f| f.name == "alpha.go")
        .expect("alpha.go not found");
    assert!(
        alpha_go.file.contains("alpha.go"),
        "alpha.go path is incorrect"
    );

    let delta_go = files
        .iter()
        .find(|f| f.name == "delta.go")
        .expect("delta.go not found");
    assert!(
        delta_go.file.contains("delta.go"),
        "delta.go path is incorrect"
    );

    /* REACT FILES */

    let tsx_files: Vec<_> = files.iter().filter(|f| f.name.ends_with(".tsx")).collect();
    assert_eq!(tsx_files.len(), 5, "Expected 5 TSX files");

    let app_tsx = files
        .iter()
        .find(|f| f.name == "App.tsx")
        .expect("App.tsx not found");
    assert!(
        app_tsx.file.contains("App.tsx"),
        "App.tsx path is incorrect"
    );

    let new_person_tsx = files
        .iter()
        .find(|f| f.name == "NewPerson.tsx")
        .expect("NewPerson.tsx not found");
    assert!(
        new_person_tsx.file.contains("NewPerson.tsx"),
        "NewPerson.tsx path is incorrect"
    );

    let people_tsx = files
        .iter()
        .find(|f| f.name == "People.tsx")
        .expect("People.tsx not found");
    assert!(
        people_tsx.file.contains("People.tsx"),
        "People.tsx path is incorrect"
    );

    let person_tsx = files
        .iter()
        .find(|f| f.name == "Person.tsx")
        .expect("Person.tsx not found");
    assert!(
        person_tsx.file.contains("Person.tsx"),
        "Person.tsx path is incorrect"
    );

    let index_tsx = files
        .iter()
        .find(|f| f.name == "index.tsx")
        .expect("index.tsx not found");
    assert!(
        index_tsx.file.contains("index.tsx"),
        "index.tsx path is incorrect"
    );

    /* PACKAGE FILES */

    let go_mod = files
        .iter()
        .find(|f| f.name == "go.mod")
        .expect("go.mod not found");
    assert!(go_mod.file.contains("go.mod"), "go.mod path is incorrect");

    let package_json = files
        .iter()
        .find(|f| f.name == "package.json")
        .expect("package.json not found");
    assert!(
        package_json.file.contains("package.json"),
        "package.json path is incorrect"
    );

    /* LIBRARIES */

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(
        libraries.len(),
        23,
        "Expected 23 libraries (5 Go + 18 React)"
    );

    /* GO LIBRARIES */

    let go_libraries: Vec<_> = libraries
        .iter()
        .filter(|l| l.file.contains("go.mod"))
        .collect();
    assert_eq!(go_libraries.len(), 5, "Expected 5 Go libraries");

    let gorm_lib = libraries
        .iter()
        .find(|l| l.name == "gorm.io/gorm")
        .expect("GORM library not found");
    assert!(
        gorm_lib.file.contains("go.mod"),
        "GORM library should be in go.mod"
    );

    let chi_lib = libraries
        .iter()
        .find(|l| l.name == "github.com/go-chi/chi")
        .expect("Chi library not found");
    assert!(
        chi_lib.file.contains("go.mod"),
        "Chi library should be in go.mod"
    );

    /* REACT LIBRARIES */

    let react_libraries: Vec<_> = libraries
        .iter()
        .filter(|l| l.file.contains("package.json"))
        .collect();
    assert_eq!(react_libraries.len(), 18, "Expected 18 React libraries");

    let react_lib = libraries
        .iter()
        .find(|l| l.name == "react")
        .expect("React library not found");
    assert!(
        react_lib.file.contains("package.json"),
        "React library should be in package.json"
    );

    let react_dom_lib = libraries
        .iter()
        .find(|l| l.name == "react-dom")
        .expect("React DOM library not found");
    assert!(
        react_dom_lib.file.contains("package.json"),
        "React DOM library should be in package.json"
    );

    let typescript_lib = libraries
        .iter()
        .find(|l| l.name == "typescript")
        .expect("TypeScript library not found");
    assert!(
        typescript_lib.file.contains("package.json"),
        "TypeScript library should be in package.json"
    );

    /* IMPORTS */

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 10, "Expected 10 import sections");

    /* GO IMPORTS */

    let go_imports: Vec<_> = imports.iter().filter(|i| i.file.ends_with(".go")).collect();
    assert_eq!(go_imports.len(), 5, "Expected 5 Go import sections");

    let main_imports = imports
        .iter()
        .find(|i| i.file.contains("main.go"))
        .expect("main.go imports not found");
    assert!(
        main_imports.body.contains("context"),
        "main.go should import context"
    );
    assert!(
        main_imports.body.contains("fmt"),
        "main.go should import fmt"
    );

    let routes_imports = imports
        .iter()
        .find(|i| i.file.contains("routes.go"))
        .expect("routes.go imports not found");
    assert!(
        routes_imports.body.contains("github.com/go-chi/chi"),
        "routes.go should import chi"
    );

    /* REACT IMPORTS */

    let react_imports: Vec<_> = imports
        .iter()
        .filter(|i| i.file.ends_with(".tsx"))
        .collect();
    assert_eq!(react_imports.len(), 4, "Expected 4 React import sections");

    let app_imports = imports
        .iter()
        .find(|i| i.file.contains("App.tsx"))
        .expect("App.tsx imports not found");
    assert!(
        app_imports.body.contains("react"),
        "App.tsx should import react"
    );
    assert!(
        app_imports.body.contains("react-router-dom"),
        "App.tsx should import react-router-dom"
    );

    /* VARIABLES */
    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 2, "Expected 2 variables");

    let db_var = variables
        .iter()
        .find(|v| v.name == "DB")
        .expect("DB variable not found");
    assert!(
        db_var.file.contains("db.go"),
        "DB variable file should contain db.go"
    );
    assert!(
        db_var.body.contains("var DB database"),
        "DB variable should have correct declaration"
    );

    let host_var = variables
        .iter()
        .find(|v| v.name == "host")
        .expect("host variable not found");
    assert!(
        host_var.file.contains("api.ts"),
        "host variable should be in api.ts"
    );

    /* CLASSES */

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 1, "Expected 1 class");

    let database_class = &classes[0];
    assert_eq!(
        database_class.name, "database",
        "Database class name is incorrect"
    );
    assert!(
        database_class.file.contains("db.go"),
        "Database class should be in db.go"
    );

    /* INSTANCES */

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 1, "Expected 1 instance");

    let db_instance = &instances[0];
    assert_eq!(db_instance.name, "DB", "DB instance name is incorrect");
    assert!(
        db_instance.file.contains("db.go"),
        "DB instance should be in db.go"
    );

    /* DATA MODELS */

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 3, "Expected 3 data models");

    let go_person_model = data_models
        .iter()
        .find(|dm| dm.name == "Person" && dm.file.contains("db.go"))
        .expect("Go Person data model not found");
    assert!(
        go_person_model.body.contains("ID"),
        "Go Person should have ID field"
    );
    assert!(
        go_person_model.body.contains("Name"),
        "Go Person should have Name field"
    );
    assert!(
        go_person_model.body.contains("Email"),
        "Go Person should have Email field"
    );

    let database_model = data_models
        .iter()
        .find(|dm| dm.name == "database")
        .expect("database data model not found");
    assert!(
        database_model.file.contains("db.go"),
        "database model should be in db.go"
    );

    let react_person_model = data_models
        .iter()
        .find(|dm| dm.name == "Person" && dm.file.contains("Person.tsx"))
        .expect("React Person data model not found");
    assert!(
        react_person_model.body.contains("id"),
        "React Person should have id field"
    );
    assert!(
        react_person_model.body.contains("name"),
        "React Person should have name field"
    );
    assert!(
        react_person_model.body.contains("email"),
        "React Person should have email field"
    );

    /* FUNCTIONS */

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    if use_lsp {
        assert_eq!(functions.len(), 26, "Expected 26 functions with LSP");
    } else {
        assert_eq!(functions.len(), 25, "Expected 25 functions ");
    }

    /* GO FUNCTIONS */

    let go_functions: Vec<_> = functions
        .iter()
        .filter(|f| f.file.ends_with(".go"))
        .collect();
    assert_eq!(go_functions.len(), 15, "Expected 15 Go functions");

    let main_fn = functions
        .iter()
        .find(|f| f.name == "main" && f.file.contains("main.go"))
        .expect("main function not found");
    assert!(main_fn.body.contains("InitDB"), "main should call InitDB");
    assert!(
        main_fn.body.contains("NewRouter"),
        "main should call NewRouter"
    );

    let init_db_fn = functions
        .iter()
        .find(|f| f.name == "InitDB")
        .expect("InitDB function not found");
    assert!(
        init_db_fn.file.contains("db.go"),
        "InitDB should be in db.go"
    );

    let new_router_fn = functions
        .iter()
        .find(|f| f.name == "NewRouter")
        .expect("NewRouter function not found");
    assert!(
        new_router_fn.file.contains("routes.go"),
        "NewRouter should be in routes.go"
    );
    assert!(
        new_router_fn.body.contains("/person"),
        "NewRouter should define /person route"
    );
    assert!(
        new_router_fn.body.contains("/people"),
        "NewRouter should define /people route"
    );

    let get_person_fn = functions
        .iter()
        .find(|f| f.name == "GetPerson")
        .expect("GetPerson function not found");
    assert!(
        get_person_fn.file.contains("routes.go"),
        "GetPerson should be in routes.go"
    );

    let create_person_fn = functions
        .iter()
        .find(|f| f.name == "CreatePerson")
        .expect("CreatePerson function not found");
    assert!(
        create_person_fn.file.contains("routes.go"),
        "CreatePerson should be in routes.go"
    );

    let get_people_fn = functions
        .iter()
        .find(|f| f.name == "GetPeople")
        .expect("GetPeople function not found");
    assert!(
        get_people_fn.file.contains("routes.go"),
        "GetPeople should be in routes.go"
    );

    let alpha_fn = functions
        .iter()
        .find(|f| f.name == "Alpha")
        .expect("Alpha function not found");
    assert!(
        alpha_fn.file.contains("alpha.go"),
        "Alpha should be in alpha.go"
    );

    let delta_fn = functions
        .iter()
        .find(|f| f.name == "Delta")
        .expect("Delta function not found");
    assert!(
        delta_fn.file.contains("delta.go"),
        "Delta should be in delta.go"
    );

    /* REACT FUNCTIONS */

    let react_functions: Vec<_> = functions
        .iter()
        .filter(|f| f.file.ends_with(".tsx"))
        .collect();

    assert_eq!(
        react_functions.len(),
        11,
        "Expected 11 React functions/components"
    );

    let app_component = functions
        .iter()
        .find(|f| f.name == "App" && f.file.contains("App.tsx"))
        .expect("App component not found");
    assert!(
        app_component.body.contains("Routes"),
        "App should use Routes"
    );
    assert!(
        app_component.body.contains("Route"),
        "App should define Route components"
    );

    let new_person_component = react_functions
        .iter()
        .find(|f| f.name == "NewPerson")
        .expect("NewPerson component not found");
    assert!(
        new_person_component.file.contains("NewPerson.tsx"),
        "NewPerson should be in NewPerson.tsx"
    );

    let people_component = react_functions
        .iter()
        .find(|f| f.name == "People")
        .expect("People component not found");
    assert!(
        people_component.file.contains("People.tsx"),
        "People component should be in People.tsx"
    );

    let styled_components = react_functions
        .iter()
        .filter(|f| {
            f.name.starts_with("Form")
                || f.name == "Label"
                || f.name == "Input"
                || f.name == "SubmitButton"
        })
        .collect::<Vec<_>>();
    assert!(
        styled_components.len() >= 6,
        "Expected at least 6 styled components"
    );

    /* ENDPOINTS */

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 3, "Expected 3 endpoints");

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/{id}" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /person/{{id}} endpoint not found");
    assert!(
        get_person_endpoint.file.contains("routes.go"),
        "GET /person/{{id}} endpoint should be in routes.go"
    );

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST /person endpoint not found");
    assert!(
        post_person_endpoint.file.contains("routes.go"),
        "POST /person endpoint should be in routes.go"
    );

    let get_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /people endpoint not found");
    assert!(
        get_people_endpoint.file.contains("routes.go"),
        "GET /people endpoint should be in routes.go"
    );

    /* REQUESTS */

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let post_request = requests
        .iter()
        .find(|r| r.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST request not found");
    assert!(
        post_request.name.contains("/person"),
        "POST request should be to /person"
    );
    assert!(
        post_request.file.contains("NewPerson.tsx"),
        "POST request should be in NewPerson.tsx"
    );

    let get_request = requests
        .iter()
        .find(|r| r.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET request not found");
    assert!(
        get_request.name.contains("/people"),
        "GET request should be to /people"
    );
    assert!(
        get_request.file.contains("People.tsx"),
        "GET request should be in People.tsx"
    );

    /* PAGES */

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes_count += pages.len();
    assert_eq!(pages.len(), 2, "Expected 2 pages");

    let home_page = pages
        .iter()
        .find(|p| p.name == "/")
        .expect("Home page not found");
    assert!(
        home_page.file.contains("App.tsx"),
        "Home page should be in App.tsx"
    );

    let new_person_page = pages
        .iter()
        .find(|p| p.name == "/new-person")
        .expect("New person page not found");
    assert!(
        new_person_page.file.contains("App.tsx"),
        "New person page should be in App.tsx"
    );

    /*****
     *
     ** EDGES TESTS **
     *
     *****/

    let contains_edges_count = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges_count;

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;
    if use_lsp {
        assert_eq!(calls_edges_count, 21, "Expected 21 calls edges with LSP");
    } else {
        assert_eq!(calls_edges_count, 22, "Expected 22 calls edges without LSP");
    }

    let renders_edges_count = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders_edges_count;

    let imports_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imports_edges_count;

    let operand_edges_count = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges_count;

    let of_edges_count = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges_count;

    /* ACCOUNT FOR ALL NODES AND EDGES */
    assert_eq!(
        num_nodes as usize, nodes_count,
        "Graph node count should match sum of all tested nodes"
    );
    assert_eq!(
        num_edges as usize, edges_count,
        "Graph edge count should match sum of all tested edges"
    );

    /*****
     *
     ** SPECIFIC TESTING **
     *
     *****/

    /* FUNCTION CALLS */

    let main_fn_node = Node::new(NodeType::Function, main_fn.clone());
    let init_db_fn_node = Node::new(NodeType::Function, init_db_fn.clone());
    let new_router_fn_node = Node::new(NodeType::Function, new_router_fn.clone());

    assert!(
        graph.has_edge(&main_fn_node, &init_db_fn_node, EdgeType::Calls),
        "main should call InitDB"
    );
    assert!(
        graph.has_edge(&main_fn_node, &new_router_fn_node, EdgeType::Calls),
        "main should call NewRouter"
    );

    let alpha_fn_node = Node::new(NodeType::Function, alpha_fn.clone());
    let delta_fn_node = Node::new(NodeType::Function, delta_fn.clone());
    assert!(
        graph.has_edge(&alpha_fn_node, &delta_fn_node, EdgeType::Calls),
        "Alpha should call Delta"
    );
    assert!(
        graph.has_edge(&delta_fn_node, &alpha_fn_node, EdgeType::Calls),
        "Delta should call Alpha"
    );

    /* HANDLER EDGES TESTS */

    let get_person_endpoint_node = Node::new(NodeType::Endpoint, get_person_endpoint.clone());
    let get_person_fn_node = Node::new(NodeType::Function, get_person_fn.clone());
    assert!(
        graph.has_edge(
            &get_person_endpoint_node,
            &get_person_fn_node,
            EdgeType::Handler
        ),
        "GET /person/{{id}} should be handled by GetPerson"
    );

    let post_person_endpoint_node = Node::new(NodeType::Endpoint, post_person_endpoint.clone());
    let create_person_fn_node = Node::new(NodeType::Function, create_person_fn.clone());
    assert!(
        graph.has_edge(
            &post_person_endpoint_node,
            &create_person_fn_node,
            EdgeType::Handler
        ),
        "POST /person should be handled by CreatePerson"
    );

    let get_people_endpoint_node = Node::new(NodeType::Endpoint, get_people_endpoint.clone());
    let get_people_fn_node = Node::new(NodeType::Function, get_people_fn.clone());
    assert!(
        graph.has_edge(
            &get_people_endpoint_node,
            &get_people_fn_node,
            EdgeType::Handler
        ),
        "GET /people should be handled by GetPeople"
    );

    /* PAGE RENDERING TESTS */

    let home_page_node = Node::new(NodeType::Page, home_page.clone());
    let people_component_node = Node::new(NodeType::Function, (*people_component).clone());
    assert!(
        graph.has_edge(&home_page_node, &people_component_node, EdgeType::Renders),
        "Home page should render People component"
    );

    let new_person_page_node = Node::new(NodeType::Page, new_person_page.clone());
    let new_person_component_node = Node::new(NodeType::Function, (*new_person_component).clone());
    assert!(
        graph.has_edge(
            &new_person_page_node,
            &new_person_component_node,
            EdgeType::Renders
        ),
        "New person page should render NewPerson component"
    );

    /* REQUEST - [CALLS] -> ENDPOINT TESTS for link_api_nodes */

    let post_request_node = Node::new(NodeType::Request, post_request.clone());
    assert!(
        graph.has_edge(
            &post_request_node,
            &post_person_endpoint_node,
            EdgeType::Calls
        ),
        "POST request should call POST /person endpoint"
    );

    let get_request_node = Node::new(NodeType::Request, get_request.clone());
    assert!(
        graph.has_edge(
            &get_request_node,
            &get_people_endpoint_node,
            EdgeType::Calls
        ),
        "GET request should call GET /people endpoint"
    );

    /* INSTANCE - OF -> CLASS */

    let database_class_node = Node::new(NodeType::Class, database_class.clone());
    let db_instance_node = Node::new(NodeType::Instance, db_instance.clone());
    assert!(
        graph.has_edge(&db_instance_node, &database_class_node, EdgeType::Of),
        "DB instance should be of database class"
    );

    let app_component_node = Node::new(NodeType::Function, app_component.clone());
    assert!(
        graph.has_edge(&app_component_node, &people_component_node, EdgeType::Calls),
        "App component should call People component"
    );
    assert!(
        graph.has_edge(
            &app_component_node,
            &new_person_component_node,
            EdgeType::Calls
        ),
        "App component should call NewPerson component"
    );
}

#[cfg(feature = "fulltest")]
#[test(tokio::test(flavor = "multi_thread", worker_threads = 3))]
async fn fulltest() {
    let use_lsp = get_use_lsp();

    let repo = Repo::new_clone_multi_detect(
        REPO_URL,
        None,
        None,
        Vec::new(),
        Vec::new(),
        Some(COMMIT),
        Some(use_lsp),
    )
    .await
    .unwrap();

    info!("Building Graphs for {} with LSP={}", REPO_URL, use_lsp);
    let array_graph = repo.build_graphs_inner::<ArrayGraph>().await.unwrap();
    let btree_graph = repo.build_graphs_inner::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use ast::lang::graphs::Neo4jGraph;

        info!(
            "Clearing and building Neo4j Graph for {} with LSP={}",
            REPO_URL, use_lsp
        );
        let neo4j_graph = Neo4jGraph::default();
        neo4j_graph.clear().await.unwrap();
        if !use_lsp {
            let neo4j_graph = repo.build_graphs_inner::<Neo4jGraph>().await.unwrap();
            fulltest_generic(&neo4j_graph, use_lsp).await;
        } else {
            info!("Skipping Neo4j test with LSP enabled to avoid hanging");
        }
    }

    fulltest_generic(&array_graph, use_lsp).await;
    fulltest_generic(&btree_graph, use_lsp).await;
}
