use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;
use crate::utils::get_use_lsp;


pub async fn test_kotlin_generic<G: Graph>() -> Result<()> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/kotlin",
        Lang::from_str("kotlin").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let mut nodes_count = 0;
    let mut edges_count = 0;

    graph.analysis();

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "kotlin",
        "Language node name should be 'kotlin'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/kotlin",
        "Language node file path is incorrect"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 repository node");

    let build_gradle_files = graph.find_nodes_by_name(NodeType::File, "build.gradle.kts");
    assert_eq!(
        build_gradle_files.len(),
        2,
        "Expected 2 build.gradle.kts files"
    );
    assert_eq!(
        build_gradle_files[0].name, "build.gradle.kts",
        "Gradle file name is incorrect"
    );

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += libraries.len();
    assert_eq!(libraries.len(), 58, "Expected 58 libraries");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 9, "Expected 9 imports");

    let main_import_body = format!(
        r#"package com.kotlintestapp.sqldelight

import android.content.Context
import app.cash.sqldelight.db.SqlDriver
import app.cash.sqldelight.driver.android.AndroidSqliteDriver
import com.kotlintestapp.db.Person
import com.kotlintestapp.db.PersonDatabase"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();
    assert_eq!(classes.len(), 8, "Expected 8 classes");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += variables.len();
    assert_eq!(variables.len(), 9, "Expected 9 variables");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        sorted_classes[3].name, "ExampleInstrumentedTest",
        "Class name is incorrect"
    );
    assert_eq!(
        normalize_path(&sorted_classes[3].file),
        "src/testing/kotlin/app/src/androidTest/java/com/kotlintestapp/ExampleInstrumentedTest.kt",
        "Class file path is incorrect"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 21, "Expected 21 functions");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    assert_eq!(data_models.len(), 3, "Expected 3 data model");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes_count += requests.len();
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let function_names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    assert!(
        function_names.contains(&"useAppContext"),
        "Should contain useAppContext function"
    );
    assert!(
        function_names.contains(&"onCreate"),
        "Should contain onCreate function"
    );
    assert!(
        function_names.contains(&"insertPerson"),
        "Should contain insertPerson function"
    );
    assert!(
        function_names.contains(&"updatePerson"),
        "Should contain updatePerson function"
    );
    assert!(
        function_names.contains(&"getAllPersons"),
        "Should contain getAllPersons function"
    );
    assert!(
        function_names.contains(&"fetchPeople"),
        "Should contain fetchPeople function"
    );
    assert!(
        function_names.contains(&"addition_isCorrect"),
        "Should contain addition_isCorrect function"
    );

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls_edges_count;
    assert_eq!(calls_edges_count, 14, "Expected 14 calls edges");

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += import_edges_count;
    assert_eq!(import_edges_count, 16, "Expected 16 import edges");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges_count += contains_edges;
    assert_eq!(contains_edges, 178, "Expected 178 contains edges");

    let handler = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handler;
    assert_eq!(handler, 0, "Expected 0 handler node");

    let operand_edges_count = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand_edges_count;
    assert_eq!(operand_edges_count, 13, "Expected 13 operand edges");

    let parentof = graph.count_edges_of_type(EdgeType::ParentOf);
    edges_count += parentof;
    assert_eq!(parentof, 1, "Expected 1 parentOf edges");

    let main_activity = classes
        .iter()
        .find(|c| {
            c.name == "MainActivity"
                && normalize_path(&c.file)
                    == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/MainActivity.kt"
        })
        .expect("MainActivity class not found");
    assert!(
        main_activity.body.contains("PersonViewModel"),
        "MainActivity should contain PersonViewModel"
    );
    assert!(
        main_activity.body.contains("ComponentActivity"),
        "MainActivity should extend ComponentActivity"
    );

    let database_helper = classes
        .iter()
        .find(|c| c.name == "DatabaseHelper")
        .expect("DatabaseHelper class not found");
    assert!(
        database_helper.body.contains("SqlDriver"),
        "DatabaseHelper should contain SqlDriver"
    );
    assert!(
        database_helper.body.contains("PersonDatabase"),
        "DatabaseHelper should contain PersonDatabase"
    );

    let example_test = classes
        .iter()
        .find(|c| c.name == "ExampleInstrumentedTest")
        .expect("ExampleInstrumentedTest class not found");
    assert!(
        example_test.body.contains("useAppContext"),
        "ExampleInstrumentedTest should contain useAppContext"
    );

    let example_unit_test = classes
        .iter()
        .find(|c| c.name == "ExampleUnitTest")
        .expect("ExampleUnitTest class not found");
    assert!(
        example_unit_test.body.contains("addition_isCorrect"),
        "ExampleUnitTest should contain addition_isCorrect"
    );

    let person_model = classes
        .iter()
        .find(|c| c.name == "Person")
        .expect("Person model not found");
    assert!(
        person_model.body.contains("data class"),
        "Person should be a data class"
    );
    assert!(
        person_model.body.contains("val owner_pubkey: String,"),
        "Person should have owner_pubkey property"
    );
    assert!(
        person_model.body.contains("val img: String,"),
        "Person should have img property"
    );

    let create_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "insertPersonsIntoDatabase"
                && normalize_path(&f.file)
                    == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/viewModels/PersonViewModel.kt"
        })
        .expect("insertPersonsIntoDatabase function not found");
    assert!(
        create_person_fn.body.contains("Dispatchers.IO"),
        "insertPersonsIntoDatabase should use Dispatchers.IO"
    );

    let get_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "fetchPeople"
                && normalize_path(&f.file)
                    == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/viewModels/PersonViewModel.kt"
        })
        .expect("fetchPeople function not found");
    assert!(
        get_person_fn.body.contains(".get()"),
        "getPerson should be a GET request"
    );
    assert!(
        get_person_fn
            .body
            .contains("Gson().fromJson(json, listType)"),
        "getPerson should use Gson for JSON parsing"
    );

    let insert_person_fn_check = functions
        .iter()
        .find(|f| f.name == "insertPerson" && normalize_path(&f.file) == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
        .expect("insertPerson function not found");
    assert!(
        insert_person_fn_check.body.contains("queries.insertPerson"),
        "insertPerson should use PersonQueries"
    );

    let update_person_fn_check = functions
        .iter()
        .find(|f| f.name == "updatePerson" && normalize_path(&f.file) == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
        .expect("updatePerson function not found");
    assert!(
        update_person_fn_check.body.contains("queries.updatePerson"),
        "updatePerson should use PersonQueries"
    );
    assert!(
        update_person_fn_check.body.contains("update"),
        "updatePerson should contain update operation"
    );

    let get_all_persons_fn = functions
        .iter()
        .find(|f| f.name == "getAllPersons")
        .expect("getAllPersons function not found");
    assert!(
        get_all_persons_fn.body.contains("selectAll"),
        "getAllPersons should use selectAll"
    );

    let delete_person_fn = functions
        .iter()
        .find(|f| f.name == "clearDatabase")
        .expect("clearDatabase function not found");
    assert!(
        delete_person_fn.body.contains("deleteAll"),
        "clearDatabase should contain deleteAll operation"
    );

    let oncreate_fn = functions
        .iter()
        .find(|f| {
            f.name == "onCreate"
                && normalize_path(&f.file)
                    == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/MainActivity.kt"
        })
        .expect("onCreate function not found");
    assert!(
        oncreate_fn.body.contains("super.onCreate"),
        "onCreate should call super.onCreate"
    );

    let use_app_context_fn = functions
        .iter()
        .find(|f| f.name == "useAppContext")
        .expect("useAppContext function not found");
    assert!(
        use_app_context_fn.body.contains("InstrumentationRegistry"),
        "useAppContext should use InstrumentationRegistry"
    );

    let addition_is_correct_fn = functions
        .iter()
        .find(|f| f.name == "addition_isCorrect")
        .expect("addition_isCorrect function not found");
    assert!(
        addition_is_correct_fn.body.contains("assertEquals"),
        "addition_isCorrect should use assertEquals"
    );

    let person_data_model = data_models
        .iter()
        .find(|dm| {
            dm.name == "Person"
                && normalize_path(&dm.file)
                    == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/models/Person.kt"
        })
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Person DataModel not found");

    let database_helper_class = classes
    .iter()
    .find(|c| c.name == "DatabaseHelper" && normalize_path(&c.file) == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
    .map(|n| Node::new(NodeType::Class, n.clone()))
    .expect("DatabaseHelper class not found");

    let insert_person_fn = functions
    .iter()
    .find(|f| f.name == "insertPerson" && normalize_path(&f.file) == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
    .map(|n| Node::new(NodeType::Function, n.clone()))
    .expect("insertPerson function not found");

    let update_person_fn = functions
    .iter()
    .find(|f| f.name == "updatePerson" && normalize_path(&f.file) == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/sqldelight/DatabaseHelper.kt")
    .map(|n| Node::new(NodeType::Function, n.clone()))
    .expect("updatePerson function not found");

    let person_kt_file = graph
        .find_nodes_by_name(NodeType::File, "Person.kt")
        .into_iter()
        .find(|n| {
            normalize_path(&n.file)
                == "src/testing/kotlin/app/src/main/java/com/kotlintestapp/models/Person.kt"
        })
        .map(|n| Node::new(NodeType::File, n))
        .expect("Person.kt file node not found");

    assert!(
        graph.has_edge(&database_helper_class, &insert_person_fn, EdgeType::Operand),
        "Expected DatabaseHelper class to operand insertPerson function"
    );

    assert!(
        graph.has_edge(&database_helper_class, &update_person_fn, EdgeType::Operand),
        "Expected DatabaseHelper class to operand updatePerson function"
    );
    assert!(
        graph.has_edge(&person_kt_file, &person_data_model, EdgeType::Contains),
        "Expected Person.kt file to contain Person DataModel"
    );

    let operand_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);

    let main_activity_operand = operand_edges
        .iter()
        .any(|(src, dst)| src.name == "MainActivity" && dst.name == "onCreate");
    assert!(
        main_activity_operand,
        "Expected MainActivity -> Operand -> onCreate edge"
    );

    let database_helper_insert_operand = operand_edges
        .iter()
        .any(|(src, dst)| src.name == "DatabaseHelper" && dst.name == "insertPerson");
    assert!(
        database_helper_insert_operand,
        "Expected DatabaseHelper -> Operand -> insertPerson edge"
    );

    let call_edges =
        graph.find_nodes_with_edge_type(NodeType::Function, NodeType::Request, EdgeType::Calls);
    assert_eq!(
        call_edges.len(),
        2,
        "Expected 2 function to request call edges"
    );

    let fetch_people_call = call_edges
        .iter()
        .any(|(src, dst)| src.name == "fetchPeople" && dst.name.contains("people"));
    assert!(
        fetch_people_call,
        "Expected fetchPeople -> Calls -> request edge"
    );

    let post_update_profile_call = call_edges
        .iter()
        .any(|(src, dst)| src.name == "postUpdateProfile" && dst.name.contains("person"));
    assert!(
        post_update_profile_call,
        "Expected postUpdateProfile -> Calls -> request edge"
    );

    let import_edges =
        graph.find_nodes_with_edge_type(NodeType::File, NodeType::Import, EdgeType::Contains);
    assert_eq!(import_edges.len(), 9, "Expected 9 file to import edges");

    let database_helper_imports = import_edges
        .iter()
        .any(|(src, _dst)| src.name == "DatabaseHelper.kt");
    assert!(
        database_helper_imports,
        "Expected DatabaseHelper.kt to have imports"
    );

    let main_activity_imports = import_edges
        .iter()
        .any(|(src, _dst)| src.name == "MainActivity.kt");
    assert!(
        main_activity_imports,
        "Expected MainActivity.kt to have imports"
    );

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 31, "Expected 31 files");

    let kotlin_files: Vec<_> = files.iter().filter(|f| f.name.ends_with(".kt")).collect();
    assert_eq!(kotlin_files.len(), 9, "Expected 9 Kotlin files");

    let gradle_files: Vec<_> = files.iter().filter(|f| f.name.contains("gradle")).collect();
    assert_eq!(gradle_files.len(), 6, "Expected 6 Gradle files");

    let manifest_files: Vec<_> = files
        .iter()
        .filter(|f| f.name == "AndroidManifest.xml")
        .collect();
    assert_eq!(
        manifest_files.len(),
        1,
        "Expected 1 AndroidManifest.xml file"
    );

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 35, "Expected 35 directories");

    let app_directory = directories
        .iter()
        .find(|d| d.name == "app")
        .expect("App directory not found");
    let app = Node::new(NodeType::Directory, app_directory.clone());

    let src_directory = directories
        .iter()
        .find(|d| d.name == "src" && d.file.contains("app/src"))
        .expect("Src directory not found");
    let src = Node::new(NodeType::Directory, src_directory.clone());

    let main_directory = directories
        .iter()
        .find(|d| d.name == "main" && d.file.contains("app/src/main"))
        .expect("Main directory not found");
    let main = Node::new(NodeType::Directory, main_directory.clone());

    let java_directory = directories
        .iter()
        .find(|d| d.name == "java" && d.file.contains("app/src/main/java"))
        .expect("Java directory not found");
    let java = Node::new(NodeType::Directory, java_directory.clone());

    let dir_relationship1 = graph.has_edge(&app, &src, EdgeType::Contains);
    assert!(
        dir_relationship1,
        "Expected Contains edge between app and src directories"
    );

    let dir_relationship2 = graph.has_edge(&src, &main, EdgeType::Contains);
    assert!(
        dir_relationship2,
        "Expected Contains edge between src and main directories"
    );

    let dir_relationship3 = graph.has_edge(&main, &java, EdgeType::Contains);
    assert!(
        dir_relationship3,
        "Expected Contains edge between main and java directories"
    );

    let person_sqldelight_file = files
        .iter()
        .find(|f| f.name == "person.sq")
        .expect("Person.sq file not found");
    assert!(
        person_sqldelight_file.body.contains("CREATE TABLE"),
        "Person.sq should contain CREATE TABLE"
    );
    assert!(
        person_sqldelight_file.body.contains("INSERT"),
        "Person.sq should contain INSERT"
    );
    assert!(
        person_sqldelight_file.body.contains("SELECT"),
        "Person.sq should contain SELECT"
    );

    let build_gradle_files: Vec<_> = files
        .iter()
        .filter(|f| f.name == "build.gradle.kts")
        .collect();
    assert!(
        build_gradle_files.len() == 2,
        "Should have at least 2 build.gradle.kts files"
    );

    let (nodes, edges) = graph.get_graph_size();
    assert_eq!(
        nodes as usize, 178,
        "Expected 178 nodes, found {} (nodes_count computed: {})",
        nodes, nodes_count
    );

    assert_eq!(
        edges as usize, 222,
        "Expected 222 edges, found {} (edges_count computed: {})",
        edges, edges_count
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_kotlin() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_kotlin_generic::<ArrayGraph>().await.unwrap();
    test_kotlin_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_kotlin_generic::<Neo4jGraph>().await.unwrap();
    }
}
