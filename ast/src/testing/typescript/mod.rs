use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{lang::Lang, repo::Repo};
use std::str::FromStr;
use test_log::test;

pub async fn test_typescript_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/typescript",
        Lang::from_str("ts").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();
    if use_lsp {
        assert_eq!(num_nodes, 52, "Expected 52 nodes");
        assert!(num_edges >= 75 && num_edges <= 77, "Expected 75 edges");
    } else {
        assert_eq!(num_nodes, 49, "Expected 49 nodes");
        assert_eq!(num_edges, 69, "Expected 69 edges");
    }

    fn normalize_path(path: &str) -> String {
        path.replace("\\", "/")
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "typescript",
        "Language node name should be 'typescript'"
    );
    assert_eq!(
        normalize_path(&language_nodes[0].file),
        "src/testing/typescript/",
        "Language node file path is incorrect"
    );

    let pkg_files = graph.find_nodes_by_name(NodeType::File, "package.json");
    assert_eq!(pkg_files.len(), 1, "Expected 1 package.json file");
    assert_eq!(
        pkg_files[0].name, "package.json",
        "Package file name is incorrect"
    );

    let imports = graph.find_nodes_by_type(NodeType::Import);

    for imp in &imports {
        let import_lines: Vec<&str> = imp
            .body
            .lines()
            .filter(|line| line.trim_start().starts_with("import "))
            .collect();

        assert!(
            import_lines.len() > 0,
            "Expected multiple import lines in {}",
            imp.file
        );
    }
    assert_eq!(imports.len(), 5, "Expected 5 imports");

    let model_import_body = format!(
        r#"import DataTypes, {{ Model }} from "sequelize";
import {{ Entity, Column, PrimaryGeneratedColumn }} from "typeorm";
import {{ sequelize }} from "./config.js";"#
    );
    let model = imports
        .iter()
        .find(|i| i.file == "src/testing/typescript/src/model.ts")
        .unwrap();

    assert_eq!(
        model.body, model_import_body,
        "Model import body is incorrect"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    if use_lsp == true {
        assert_eq!(functions.len(), 9, "Expected 9 functions");
    } else {
        assert_eq!(functions.len(), 6, "Expected 6 functions");
    }

    let requests = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(requests.len(), 2, "Expected 2 requests");

    let calls_edges_count = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(calls_edges_count, 2, "Expected 2 calls edges");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 4, "Expected 4 data models");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    assert_eq!(variables.len(), 4, "Expected 4 variables");

    let contains = graph.count_edges_of_type(EdgeType::Contains);

    if use_lsp {
        assert!(
            contains >= 51 && contains <= 53,
            "Expected 51 contains edges"
        );
    } else {
        assert_eq!(contains, 53, "Expected 53 contains edges");
    }

    let import_edges_count = graph.count_edges_of_type(EdgeType::Imports);
    if use_lsp {
        assert_eq!(import_edges_count, 15, "Expected 15 import edges");
    } else {
        assert_eq!(import_edges_count, 12, "Expected 12 import edges");
    }

    let create_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "createPerson"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("createPerson function not found");

    let get_person_fn = functions
        .iter()
        .find(|f| {
            f.name == "getPerson"
                && normalize_path(&f.file) == "src/testing/typescript/src/routes.ts"
        })
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("getPerson function not found");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/person"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found");

    assert!(
        graph.has_edge(&post_person_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected '/person' POST endpoint to be handled by createPerson"
    );

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| {
            e.name == "/person/:id"
                && normalize_path(&e.file) == "src/testing/typescript/src/routes.ts"
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /person/:id endpoint not found");

    assert!(
        graph.has_edge(&get_person_endpoint, &get_person_fn, EdgeType::Handler),
        "Expected '/person/:id' GET endpoint to be handled by getPerson"
    );
    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_typescript() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_typescript_generic::<BTreeMapGraph>().await.unwrap();
    test_typescript_generic::<ArrayGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let mut graph = Neo4jGraph::new();
        graph.clear().await.unwrap();
        test_typescript_generic::<Neo4jGraph>().await.unwrap();
    }
}
