#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;

use ast::lang::graphs::{BTreeMapGraph, EdgeType};
use ast::lang::{Graph, NodeType};
use ast::repo::{clone_repo, Repo};
use lsp::git::get_changed_files_between;
use tracing::info;

const REPO_URL: &str = "https://github.com/fayekelmith/demorepo.git";
const BEFORE_COMMIT: &str = "3a2bd5cc2e0a38ce80214a32ed06b2fb9430ab73";
const AFTER_COMMIT: &str = "778b5202fca04a2cd5daed377c0063e9af52b24c";
const USE_LSP: Option<bool> = Some(false);

fn assert_common<G: Graph>(graph: &G, phase: &str) {
    let (num_nodes, num_edges) = graph.get_graph_size();
    info!("[{}] Nodes: {}, Edges: {}", phase, num_nodes, num_edges);
    assert!(num_nodes >= 100, "[{}] Expected >= 100 nodes, got {}", phase, num_nodes);
    assert!(num_edges >= 140, "[{}] Expected >= 140 edges, got {}", phase, num_edges);

    // Classes
    let classes = graph.find_nodes_by_type(NodeType::Class);
    assert_eq!(classes.len(), 2, "[{}] Expected 2 classes", phase);
    assert!(classes.iter().any(|c| c.name == "database"), "[{}] Missing class database", phase);
    assert!(classes.iter().any(|c| c.name == "Person"), "[{}] Missing class Person", phase);

    // Core functions
    let functions = graph.find_nodes_by_type(NodeType::Function);
    for name in ["main", "Alpha", "InitDB", "NewRouter", "initChi",
                  "GetPerson", "CreatePerson", "GetPeople", "GetPersonById",
                  "GetAllPeople", "CreateOrEditPerson", "UpdatePersonName",
                  "TableName", "NewPerson", "App", "People"] {
        assert!(
            functions.iter().any(|f| f.name == name),
            "[{}] Missing function {}", phase, name
        );
    }

    // Endpoints
    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 3, "[{}] Expected 3 endpoints", phase);
    let get_person = endpoints.iter().find(|e| e.name == "/person/{id}").expect(&format!("[{}] Missing GET /person/{{id}}", phase));
    assert_eq!(get_person.meta.get("verb").map(|v| v.as_str()), Some("GET"), "[{}] /person/{{id}} should be GET", phase);
    let post_person = endpoints.iter().find(|e| e.name == "/person").expect(&format!("[{}] Missing POST /person", phase));
    assert_eq!(post_person.meta.get("verb").map(|v| v.as_str()), Some("POST"), "[{}] /person should be POST", phase);
    let get_people = endpoints.iter().find(|e| e.name == "/people").expect(&format!("[{}] Missing GET /people", phase));
    assert_eq!(get_people.meta.get("verb").map(|v| v.as_str()), Some("GET"), "[{}] /people should be GET", phase);

    // Handler edges: endpoint -> function
    let handlers = graph.find_nodes_with_edge_type(NodeType::Endpoint, NodeType::Function, EdgeType::Handler);
    assert_eq!(handlers.len(), 3, "[{}] Expected 3 handler edges", phase);
    assert!(handlers.iter().any(|(e, f)| e.name == "/people" && f.name == "GetPeople"), "[{}] Missing /people -> GetPeople handler", phase);
    assert!(handlers.iter().any(|(e, f)| e.name == "/person" && f.name == "CreatePerson"), "[{}] Missing /person -> CreatePerson handler", phase);
    assert!(handlers.iter().any(|(e, f)| e.name == "/person/{id}" && f.name == "GetPerson"), "[{}] Missing /person/{{id}} -> GetPerson handler", phase);

    // DataModels
    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    assert_eq!(data_models.len(), 3, "[{}] Expected 3 data models", phase);
    assert!(data_models.iter().any(|d| d.name == "Person" && d.file.ends_with("db.go")), "[{}] Missing DataModel Person (Go)", phase);
    assert!(data_models.iter().any(|d| d.name == "database"), "[{}] Missing DataModel database", phase);

    // Requests (frontend -> backend)
    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(requests.len(), 2, "[{}] Expected 2 requests", phase);
    let req_people = requests.iter().find(|r| r.meta.get("verb").map(|v| v.as_str()) == Some("GET")).expect(&format!("[{}] Missing GET request", phase));
    assert!(req_people.name.contains("/people"), "[{}] GET request should target /people", phase);
    let req_person = requests.iter().find(|r| r.meta.get("verb").map(|v| v.as_str()) == Some("POST")).expect(&format!("[{}] Missing POST request", phase));
    assert!(req_person.name.contains("/person"), "[{}] POST request should target /person", phase);

    // Request -> Endpoint calls (cross-repo linking)
    let req_ep_calls = graph.find_nodes_with_edge_type(NodeType::Request, NodeType::Endpoint, EdgeType::Calls);
    assert_eq!(req_ep_calls.len(), 2, "[{}] Expected 2 request->endpoint call edges", phase);

    // Pages (React routes)
    let pages = graph.find_nodes_by_type(NodeType::Page);
    assert_eq!(pages.len(), 2, "[{}] Expected 2 pages", phase);
    assert!(pages.iter().any(|p| p.name == "/"), "[{}] Missing page /", phase);
    assert!(pages.iter().any(|p| p.name == "/new-person"), "[{}] Missing page /new-person", phase);

    // Operand edges (class -> method)
    let operand_count = graph.count_edges_of_type(EdgeType::Operand);
    assert_eq!(operand_count, 6, "[{}] Expected 6 operand edges", phase);

    // Stable call edges
    let calls = graph.find_nodes_with_edge_type(NodeType::Function, NodeType::Function, EdgeType::Calls);
    for (src, tgt) in [("main", "InitDB"), ("main", "NewRouter"), ("GetPerson", "GetPersonById"),
                       ("GetPeople", "GetAllPeople"), ("CreatePerson", "NewPerson")] {
        assert!(
            calls.iter().any(|(s, t)| s.name == src && t.name == tgt),
            "[{}] Missing call edge {} -> {}", phase, src, tgt
        );
    }
}

fn assert_before<G: Graph>(graph: &G) {
    let phase = "BEFORE";
    assert_common(graph, phase);

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert!(functions.iter().any(|f| f.name == "Beta"), "[{}] Missing function Beta", phase);
    assert!(functions.iter().any(|f| f.name == "Gamma"), "[{}] Missing function Gamma", phase);
    assert!(
        !functions.iter().any(|f| f.name == "Delta"),
        "[{}] Delta should not exist in BEFORE", phase
    );

    let calls = graph.find_nodes_with_edge_type(NodeType::Function, NodeType::Function, EdgeType::Calls);
    assert!(calls.iter().any(|(s, t)| s.name == "Alpha" && t.name == "Beta"), "[{}] Missing Alpha -> Beta", phase);
    assert!(calls.iter().any(|(s, t)| s.name == "Alpha" && t.name == "Gamma"), "[{}] Missing Alpha -> Gamma", phase);
    assert!(calls.iter().any(|(s, t)| s.name == "Beta" && t.name == "Alpha"), "[{}] Missing Beta -> Alpha", phase);

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 105, "[{}] Expected 105 nodes", phase);
    assert_eq!(num_edges, 153, "[{}] Expected 153 edges", phase);
}

fn assert_after<G: Graph>(graph: &G) {
    let phase = "AFTER";
    assert_common(graph, phase);

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert!(functions.iter().any(|f| f.name == "Delta"), "[{}] Missing function Delta", phase);
    assert!(
        !functions.iter().any(|f| f.name == "Beta"),
        "[{}] Beta should not exist in AFTER", phase
    );
    assert!(
        !functions.iter().any(|f| f.name == "Gamma"),
        "[{}] Gamma should not exist in AFTER", phase
    );

    let calls = graph.find_nodes_with_edge_type(NodeType::Function, NodeType::Function, EdgeType::Calls);
    assert!(calls.iter().any(|(s, t)| s.name == "Alpha" && t.name == "Delta"), "[{}] Missing Alpha -> Delta", phase);
    assert!(calls.iter().any(|(s, t)| s.name == "Delta" && t.name == "Alpha"), "[{}] Missing Delta -> Alpha", phase);
    assert!(
        !calls.iter().any(|(s, t)| s.name == "Alpha" && t.name == "Beta"),
        "[{}] Alpha -> Beta should not exist in AFTER", phase
    );

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(num_nodes, 104, "[{}] Expected 104 nodes", phase);
    assert_eq!(num_edges, 151, "[{}] Expected 151 edges", phase);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_graph_accuracy() {
    let repo_path = Repo::get_path_from_url(REPO_URL).unwrap();

    clone_repo(REPO_URL, &repo_path, None, None, Some(BEFORE_COMMIT), None)
        .await
        .unwrap();

    let repos = Repo::new_multi_detect(
        &repo_path,
        Some(REPO_URL.to_string()),
        Vec::new(),
        Vec::new(),
        USE_LSP,
    )
    .await
    .unwrap();

    let btree_graph = repos.build_graphs_inner::<BTreeMapGraph>().await.unwrap();
    assert_before(&btree_graph);

    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops.clear().await.unwrap();
        graph_ops
            .update_full(
                REPO_URL,
                None,
                None,
                BEFORE_COMMIT,
                Some(BEFORE_COMMIT),
                None,
                USE_LSP,
                Some(false),
            )
            .await
            .unwrap();
        assert_before(&graph_ops.graph);
    }

    clone_repo(REPO_URL, &repo_path, None, None, Some(AFTER_COMMIT), None)
        .await
        .unwrap();

    let changed_files = get_changed_files_between(&repo_path, BEFORE_COMMIT, AFTER_COMMIT)
        .await
        .unwrap();

    let expected_files = ["alpha.go", "beta.go", "delta.go"];
    for file in expected_files {
        assert!(
            changed_files.contains(&file.to_string()),
            "Expected changed file {} not found",
            file
        );
    }

    let new_repos = Repo::new_multi_detect(
        &repo_path,
        Some(REPO_URL.to_string()),
        Vec::new(),
        Vec::new(),
        USE_LSP,
    )
    .await
    .unwrap();

    let new_btree_graph = new_repos
        .build_graphs_inner::<BTreeMapGraph>()
        .await
        .unwrap();

    assert_after(&new_btree_graph);

    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops
            .update_incremental(
                REPO_URL,
                None,
                None,
                AFTER_COMMIT,
                BEFORE_COMMIT,
                Some(AFTER_COMMIT),
                None,
                USE_LSP,
                None,
            )
            .await
            .unwrap();

        graph_ops.graph.analysis();
        assert_after(&graph_ops.graph);

        let (btree_nodes, btree_edges) = new_btree_graph.get_graph_size();
        let (neo4j_nodes, neo4j_edges) = graph_ops.graph.get_graph_size();

        let (btree_node_keys, btree_edge_keys) = new_btree_graph.get_graph_keys();
        let (neo4j_node_keys, neo4j_edge_keys) = graph_ops.graph.get_graph_keys();

        let btree_nodes_only: Vec<_> = btree_node_keys.difference(&neo4j_node_keys).collect();
        let neo4j_nodes_only: Vec<_> = neo4j_node_keys.difference(&btree_node_keys).collect();

        let btree_only: Vec<_> = btree_edge_keys.difference(&neo4j_edge_keys).collect();
        let neo4j_only: Vec<_> = neo4j_edge_keys.difference(&btree_edge_keys).collect();

        assert!(
            btree_only.is_empty() && neo4j_only.is_empty(),
            "Edge key mismatch between BTreeMapGraph and Neo4jGraph"
        );

        assert!(
            btree_nodes_only.is_empty() && neo4j_nodes_only.is_empty(),
            "Node key mismatch between BTreeMapGraph and Neo4jGraph"
        );

        assert_eq!(
            btree_nodes, neo4j_nodes,
            "BTreeMapGraph and Neo4jGraph node count mismatch. BTreeMap only: {:?}, Neo4j only: {:?}",
            btree_nodes_only, neo4j_nodes_only
        );

        assert_eq!(
            btree_edges, neo4j_edges,
            "BTreeMapGraph and Neo4jGraph edge count mismatch. BTreeMap only: {:?}, Neo4j only: {:?}",
            btree_only, neo4j_only
        );
    }
}
