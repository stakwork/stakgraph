use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;
use test_log::test;

pub async fn test_ruby_generic<G: Graph>() -> Result<()> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/ruby",
        Lang::from_str("ruby").unwrap(),
        use_lsp,
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
        language_nodes[0].name, "ruby",
        "Language node name should be 'ruby'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/ruby",
        "Language node file path is incorrect"
    );

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();

    if use_lsp {
        let expected = 42;
        assert!(
            (expected - 1..=expected + 1).contains(&files.len()),
            "Expected ~{} file nodes with LSP, got {}",
            expected,
            files.len()
        );
    } else {
        assert_eq!(
            files.len(),
            42,
            "Expected 42 file nodes, got {}",
            files.len()
        );
    }

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(
        libraries.len(),
        5,
        "Expected 5 library nodes, got {}",
        libraries.len()
    );

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "Gemfile");
    assert_eq!(pkg_files.len(), 1, "Expected 1 Gemfile");
    assert_eq!(
        pkg_files[0].name, "Gemfile",
        "Package file name is incorrect"
    );
    assert!(
        pkg_files[0].body.contains("rails"),
        "Gemfile should contain rails gem"
    );
    assert!(
        pkg_files[0].body.contains("sqlite3"),
        "Gemfile should contain sqlite3 gem"
    );
    assert!(
        pkg_files[0].body.contains("puma"),
        "Gemfile should contain puma gem"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(
        imports.len(),
        13,
        "Expected 13 import nodes, got {}",
        imports.len()
    );

    let import_body = imports
        .iter()
        .find(|i| i.file == "src/testing/ruby/config/environment.rb")
        .expect("Import body not found");
    let environment_body = format!(r#"require_relative "application""#,);

    assert_eq!(
        import_body.body, environment_body,
        "Import body is incorrect"
    );

    let boot_import = imports
        .iter()
        .find(|i| i.file == "src/testing/ruby/config/boot.rb")
        .expect("Boot import not found");
    assert!(
        boot_import.body.contains("require \"bundler/setup\""),
        "Boot should require bundler/setup"
    );

    let application_import = imports
        .iter()
        .find(|i| i.file == "src/testing/ruby/config/application.rb")
        .expect("Application import not found");
    assert!(
        application_import
            .body
            .contains("require_relative \"boot\""),
        "Application should require boot"
    );
    assert!(
        application_import.body.contains("require \"rails\""),
        "Application should require rails"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 7, "Expected 7 endpoints");

    let mut sorted_endpoints = endpoints.clone();
    sorted_endpoints.sort_by(|a, b| a.name.cmp(&b.name));

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person/:id" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET person/:id endpoint not found");
    assert_eq!(
        get_person_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST person endpoint not found");
    assert_eq!(
        post_person_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let delete_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/:id" && e.meta.get("verb") == Some(&"DELETE".to_string()))
        .expect("DELETE /people/:id endpoint not found");
    assert_eq!(
        delete_people_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let get_articles_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people/articles" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET /people/articles endpoint not found");
    assert_eq!(
        get_articles_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_articles_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/people/:id/articles" && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /people/:id/articles endpoint not found");
    assert_eq!(
        post_articles_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let post_countries_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/countries/:country_id/process"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .expect("POST /countries/:country_id/process endpoint not found");
    assert_eq!(
        post_countries_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let get_profile_endpoint = endpoints
        .iter()
        .find(|e| e.name == "profile" && e.meta.get("verb") == Some(&"GET".to_string()))
        .expect("GET profile endpoint not found");
    assert_eq!(
        get_profile_endpoint.file, "src/testing/ruby/config/routes.rb",
        "Endpoint file path is incorrect"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    //var is not in a .rb file, so it is not detected
    assert_eq!(variables.len(), 0, "Expected 0 variable nodes");

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler_edges_count;
    assert_eq!(handler_edges_count, 7, "Expected 7 handler edges");

    let class_counts = graph.count_edges_of_type(EdgeType::ParentOf);
    edges_count += class_counts;
    assert_eq!(class_counts, 6, "Expected 6 class edges");

    let class_calls =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Class, EdgeType::Calls);

    assert_eq!(class_calls.len(), 1, "Expected 1 class calls edges");

    let import_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges;
    if use_lsp {
        assert_eq!(import_edges, 0, "Expected 0 import edges with lsp");
    } else {
        assert_eq!(import_edges, 4, "Expected 4 import edges");
    }

    let imports_edges =
        graph.find_nodes_with_edge_type(NodeType::File, NodeType::Class, EdgeType::Imports);
    for (imp_src, imp_target) in imports_edges {
        println!("imp_edge: {} -> {}", imp_src.name, imp_target.name);
    }

    let person_to_article_call = class_calls.iter().any(|(src, dst)| {
        (src.name == "Person" && dst.name == "Article")
            || (src.name == "Article" && dst.name == "Person")
    });
    assert!(
        person_to_article_call,
        "Expects a Person -> CALLS -> Article Class Call Edge"
    );

    let contains_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::DataModel, EdgeType::Contains);

    assert_eq!(contains_edges.len(), 2, "Expected 2 contains edges");

    let person_contains_data_model = contains_edges
        .iter()
        .any(|(src, dst)| src.name == "PeopleController" && dst.name == "people");
    assert!(
        person_contains_data_model,
        "Expects a PeopleController -> CONTAINS -> people Data Model Edge"
    );

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls;

    if use_lsp {
        assert_eq!(calls, 30, "Expected 30 call edges with lsp");
    } else {
        assert_eq!(calls, 17, "Expected 17 call edges");
    }

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    if use_lsp {
        assert_eq!(uses, 14, "Expected 14 Uses edges, got {}", uses);
    } else {
        assert_eq!(uses, 0, "Expected 0 Uses edges, got {}", uses);
    }

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains;
    assert_eq!(
        contains, 138,
        "Expected 138 Contains edges, got {}",
        contains
    );

    let renders = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders;
    assert_eq!(renders, 1, "Expected 1 render edge");

    let operands = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operands;
    assert_eq!(operands, 18, "Expected 18 operand edges, got {}", operands);

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(
        classes.len(),
        16,
        "Expected 16 class nodes, got {}",
        classes.len()
    );
    let person_model = classes
        .iter()
        .find(|c| c.name == "Person" && c.file.ends_with("app/models/person.rb"))
        .expect("Person model not found");
    assert!(
        person_model.body.contains("has_many :articles"),
        "Person should have many articles"
    );
    assert!(
        person_model
            .body
            .contains("validates :name, presence: true"),
        "Person should validate name presence"
    );
    assert!(
        person_model
            .body
            .contains("validates :email, presence: true, uniqueness: true"),
        "Person should validate email presence and uniqueness"
    );

    let article_model = classes
        .iter()
        .find(|c| c.name == "Article" && c.file.ends_with("app/models/article.rb"))
        .expect("Article model not found");
    assert!(
        article_model.body.contains("belongs_to :person"),
        "Article should belong to person"
    );
    assert!(
        article_model
            .body
            .contains("validates :title, presence: true"),
        "Article should validate title presence"
    );
    assert!(
        article_model
            .body
            .contains("validates :body, presence: true"),
        "Article should validate body presence"
    );

    let country_model = classes
        .iter()
        .find(|c| c.name == "Country" && c.file.ends_with("app/models/country.rb"))
        .expect("Country model not found");
    assert!(
        country_model
            .body
            .contains("validates :name, presence: true"),
        "Country should validate name presence"
    );
    assert!(
        country_model
            .body
            .contains("validates :code, presence: true, uniqueness: true"),
        "Country should validate code presence and uniqueness"
    );

    let app_record = classes
        .iter()
        .find(|c| {
            c.name == "ApplicationRecord" && c.file.ends_with("app/models/application_record.rb")
        })
        .expect("ApplicationRecord not found");
    assert!(
        app_record.body.contains("primary_abstract_class"),
        "ApplicationRecord should have primary_abstract_class"
    );

    let app_controller = classes
        .iter()
        .find(|c| {
            c.name == "ApplicationController"
                && c.file
                    .ends_with("app/controllers/application_controller.rb")
        })
        .expect("ApplicationController not found");
    assert!(
        app_controller.body.contains("ActionController::API"),
        "ApplicationController should inherit from ActionController::API"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    let get_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "get_person" && f.file.ends_with("app/controllers/people_controller.rb")
        })
        .expect("get_person method not found");
    assert!(
        get_person_fn
            .body
            .contains("PersonService.get_person_by_id"),
        "get_person should call PersonService.get_person_by_id"
    );
    assert!(
        get_person_fn.body.contains("render json: person"),
        "get_person should render json response"
    );

    let create_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "create_person" && f.file.ends_with("app/controllers/people_controller.rb")
        })
        .expect("create_person method not found");
    assert!(
        create_person_fn.body.contains("PersonService.new_person"),
        "create_person should call PersonService.new_person"
    );
    assert!(
        create_person_fn.body.contains("person_params"),
        "create_person should use person_params"
    );

    let destroy_fn = functions
        .iter()
        .find(|f| f.name == "destroy" && f.file.ends_with("app/controllers/people_controller.rb"))
        .expect("destroy method not found");
    assert!(
        destroy_fn.body.contains("PersonService.delete"),
        "destroy should call PersonService.delete"
    );

    let articles_fn = functions
        .iter()
        .find(|f| f.name == "articles" && f.file.ends_with("app/controllers/people_controller.rb"))
        .expect("articles method not found");
    assert!(
        articles_fn.body.contains("Article.all"),
        "articles should call Article.all"
    );

    let show_profile_fn = functions
        .iter()
        .find(|f| {
            f.name == "show_person_profile"
                && f.file.ends_with("app/controllers/people_controller.rb")
        })
        .expect("show_person_profile method not found");
    assert!(
        show_profile_fn.body.contains("Person.find(params[:id])"),
        "show_person_profile should find person by id"
    );

    let process_fn = functions
        .iter()
        .find(|f| {
            f.name == "process" && f.file.ends_with("app/controllers/countries_controller.rb")
        })
        .expect("process method not found");
    assert!(
        process_fn.body.contains("Country.new(country_params)"),
        "process should create new Country"
    );

    let get_person_by_id = functions
        .iter()
        .find(|f| {
            f.name == "get_person_by_id" && f.file.ends_with("app/services/person_service.rb")
        })
        .expect("get_person_by_id method not found");
    assert!(
        get_person_by_id.body.contains("Person.find_by(id: id)"),
        "get_person_by_id should call Person.find_by"
    );

    let new_person = functions
        .iter()
        .find(|f| f.name == "new_person" && f.file.ends_with("app/services/person_service.rb"))
        .expect("new_person method not found");
    assert!(
        new_person.body.contains("Person.create(person_params)"),
        "new_person should call Person.create"
    );

    let delete = functions
        .iter()
        .find(|f| f.name == "delete" && f.file.ends_with("app/services/person_service.rb"))
        .expect("delete method not found");
    assert!(
        delete.body.contains("Person.destroy(id)"),
        "delete should call Person.destroy"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    let people_table = data_models
        .iter()
        .find(|dm| dm.name == "people" && dm.file.ends_with("db/schema.rb"))
        .expect("people DataModel not found");
    assert!(
        people_table.body.contains("t.string \"name\""),
        "people table should have name column"
    );
    assert!(
        people_table.body.contains("t.string \"email\""),
        "people table should have email column"
    );
    assert!(
        people_table
            .body
            .contains("index [\"email\"], name: \"index_people_on_email\", unique: true"),
        "people table should have unique email index"
    );

    let articles_table = data_models
        .iter()
        .find(|dm| dm.name == "articles" && dm.file.ends_with("db/schema.rb"))
        .expect("articles DataModel not found");
    assert!(
        articles_table.body.contains("t.string \"title\""),
        "articles table should have title column"
    );
    assert!(
        articles_table.body.contains("t.text \"body\""),
        "articles table should have body column"
    );
    assert!(
        articles_table
            .body
            .contains("t.integer \"person_id\", null: false"),
        "articles table should have person_id foreign key"
    );

    let create_people_migration = classes
        .iter()
        .find(|c| c.name == "CreatePeople" && c.file.contains("create_people.rb"))
        .expect("CreatePeople migration not found");
    assert!(
        create_people_migration
            .body
            .contains("create_table :people"),
        "CreatePeople should create people table"
    );

    let create_articles_migration = classes
        .iter()
        .find(|c| c.name == "CreateArticles" && c.file.contains("create_articles.rb"))
        .expect("CreateArticles migration not found");
    assert!(
        create_articles_migration
            .body
            .contains("create_table :articles"),
        "CreateArticles should create articles table"
    );
    assert!(
        create_articles_migration
            .body
            .contains("t.references :person, null: false, foreign_key: true"),
        "CreateArticles should add person foreign key"
    );

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes_count += pages.len();
    assert_eq!(pages.len(), 1, "Expected 1 page");
    let profile_page = &pages[0];
    assert_eq!(
        profile_page.name, "show_person_profile.html.erb",
        "Page name should be show_person_profile.html.erb"
    );
    assert!(
        profile_page.file.ends_with("show_person_profile.html.erb"),
        "Page should be an erb file"
    );
    assert!(
        profile_page.body.contains("@person.name"),
        "Profile page should display person name"
    );

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(
        directories.len(),
        22,
        "Expected 22 directories, got {}",
        directories.len()
    );

    let app_directory = directories
        .iter()
        .find(|d| d.name == "app" && d.file.ends_with("src/testing/ruby/app"))
        .expect("App directory not found");
    let app = Node::new(NodeType::Directory, app_directory.clone());

    let controllers_directory = directories
        .iter()
        .find(|d| d.name == "controllers" && d.file.ends_with("src/testing/ruby/app/controllers"))
        .expect("Controllers directory not found");
    let controllers = Node::new(NodeType::Directory, controllers_directory.clone());

    let dir_relationship = graph.has_edge(&app, &controllers, EdgeType::Contains);
    assert!(
        dir_relationship,
        "Expected Contains edge between app and controllers directories"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    assert_eq!(
        unit_tests.len(),
        3,
        "Expected 3 unit tests, got {}",
        unit_tests.len()
    );
    nodes_count += unit_tests.len();
    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    assert_eq!(
        integration_tests.len(),
        2,
        "Expected 2 integration tests, got {}",
        integration_tests.len()
    );
    nodes_count += integration_tests.len();
    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    assert_eq!(
        e2e_tests.len(),
        6,
        "Expected 6 e2e tests, got {}",
        e2e_tests.len()
    );
    nodes_count += e2e_tests.len();

    let (nodes, edges) = graph.get_graph_size();

    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes",
        nodes_count
    );

    assert_eq!(
        edges as usize, edges_count,
        "Expected {} edges, found {} (edges_count computed: {})",
        edges_count, edges, edges_count
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_ruby() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_ruby_generic::<ArrayGraph>().await.unwrap();
    test_ruby_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_ruby_generic::<Neo4jGraph>().await.unwrap();
    }
}
