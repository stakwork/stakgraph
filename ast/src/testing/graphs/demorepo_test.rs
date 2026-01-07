use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::BTreeMapGraph;
use crate::lang::{Graph, Node};
use crate::repo::{Repo, Repos};
use crate::utils::get_use_lsp;
use shared::error::Result;
use test_log::test;

async fn test_demorepo_generic<G: Graph>(repos: &Repos) -> Result<()> {
    let use_lsp = get_use_lsp();
    let graph = repos.build_graphs_inner::<BTreeMapGraph>().await.unwrap();

    graph.analysis();

    let (num_nodes, num_edges) = graph.get_graph_size();

    if use_lsp {
        assert_eq!(num_nodes, 145, "Expected 145 nodes in the graph");
        assert_eq!(num_edges, 216, "Expected 216 edges in the graph");
    } else {
        assert_eq!(num_nodes, 102, "Expected 102 nodes in the graph");
        assert_eq!(num_edges, 144, "Expected 144 edges in the graph");
    }

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 2, "Expected 2 language nodes");

    language_nodes.iter().for_each(|node| {
        assert!(
            matches!(node.name.as_str(), "go" | "react"),
            "Unexpected language node: {:?}",
            node
        );
    });

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    assert_eq!(endpoints.len(), 3, "Expected 3 endpoint nodes");
    assert!(
        endpoints.iter().any(|e| e.name == "/person"),
        "Expected at least one endpoint containing 'person'"
    );
    let functions = graph.find_nodes_by_type(NodeType::Function);
    if use_lsp {
        assert_eq!(functions.len(), 69, "Expected 69 function nodes");
    } else {
        assert_eq!(functions.len(), 26, "Expected 26 function nodes");
    }
    assert!(
        functions
            .iter()
            .any(|f| f.name == "NewRouter" && f.file.ends_with("routes.go")),
        "Function 'NewRouter' not found"
    );

    let requests = graph.find_nodes_by_type(NodeType::Request);
    assert_eq!(
        requests.len(),
        2,
        "Expected at least one Request node (React/TSX frontend)"
    );
    assert!(
        requests.iter().any(|r| r.name == "${api.host}/person"),
        "Expected at least one Request node for 'GET /person'"
    );

    let get_people_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/people" && e.meta.get("verb") == Some(&"GET".to_string()))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /people endpoint not found");

    let post_person_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /person endpoint not found");

    let get_people_request = requests
        .iter()
        .find(|r| {
            r.name == "${api.host}/people"
                && r.meta.get("verb") == Some(&"GET".to_string())
                && r.file.ends_with("frontend/src/components/People.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("GET request to ${api.host}/people not found");

    let post_person_request = requests
        .iter()
        .find(|r| {
            r.name == "${api.host}/person"
                && r.meta.get("verb") == Some(&"POST".to_string())
                && r.file.ends_with("frontend/src/components/NewPerson.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("POST request to ${api.host}/person not found");

    let get_people_handler_func = functions
        .iter()
        .find(|f| f.name == "GetPeople" && f.file.ends_with("routes.go"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("GET handler function for people not found");

    let post_person_handler_func = functions
        .iter()
        .find(|f| f.name == "CreatePerson" && f.file.ends_with("routes.go"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("POST handler function for person not found");

    let people_component_func = functions
        .iter()
        .find(|f| f.name == "People" && f.file.ends_with("People.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'People' component not found");

    let new_person_component_func = functions
        .iter()
        .find(|f| f.name == "NewPerson" && f.file.ends_with("NewPerson.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'NewPerson' component not found");

    // GET /people: Request → Endpoint
    assert!(
        graph.has_edge(&get_people_request, &get_people_endpoint, EdgeType::Calls),
        "Expected GET people request to call the GET /people endpoint"
    );

    // POST /person: Request → Endpoint
    assert!(
        graph.has_edge(&post_person_request, &post_person_endpoint, EdgeType::Calls),
        "Expected POST person request to call the POST /person endpoint"
    );

    assert!(
        graph.has_edge(
            &get_people_endpoint,
            &get_people_handler_func,
            EdgeType::Handler
        ),
        "Expected GET /people endpoint to be handled by GetPeople function"
    );

    assert!(
        graph.has_edge(
            &post_person_endpoint,
            &post_person_handler_func,
            EdgeType::Handler
        ),
        "Expected POST /person endpoint to be handled by CreatePerson function"
    );

    assert!(
        graph.has_edge(&people_component_func, &get_people_request, EdgeType::Calls),
        "Expected People component to call the GET /people request"
    );

    assert!(
        graph.has_edge(
            &new_person_component_func,
            &post_person_request,
            EdgeType::Calls
        ),
        "Expected NewPerson component to call the POST /person request"
    );

    assert!(
        graph.has_edge(&people_component_func, &get_people_request, EdgeType::Calls),
        "People component → GET people request"
    );
    assert!(
        graph.has_edge(&get_people_request, &get_people_endpoint, EdgeType::Calls),
        "GET people request → GET people endpoint"
    );
    assert!(
        graph.has_edge(
            &get_people_endpoint,
            &get_people_handler_func,
            EdgeType::Handler
        ),
        "GET people endpoint → GetPeople handler"
    );
    assert!(
        graph.has_edge(
            &new_person_component_func,
            &post_person_request,
            EdgeType::Calls
        ),
        "NewPerson component → POST person request"
    );
    assert!(
        graph.has_edge(&post_person_request, &post_person_endpoint, EdgeType::Calls),
        "POST person request → POST person endpoint"
    );
    assert!(
        graph.has_edge(
            &post_person_endpoint,
            &post_person_handler_func,
            EdgeType::Handler
        ),
        "POST person endpoint → CreatePerson handler"
    );

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    assert_eq!(
        calls, 22,
        "Expected 22 edges of type Calls, found {}",
        calls
    );
    let contains = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(
        contains, 107,
        "Expected 107 edges of type Contains, found {}",
        contains
    );

    Ok(())
}

#[test(tokio::test(flavor = "multi_thread", worker_threads = 2))]
async fn test_demorepo() {
    let repo_url = "https://github.com/fayekelmith/demorepo";
    let use_lsp = Some(get_use_lsp());

    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        use_lsp,
    )
    .await
    .unwrap();

    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_demorepo_generic::<ArrayGraph>(&repos).await.unwrap();
    test_demorepo_generic::<BTreeMapGraph>(&repos)
        .await
        .unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        std::env::set_var("NEO4J_TEST_NAMESPACE", "test_demorepo");
        let graph = Neo4jGraph::with_namespace("test_demorepo");
        graph.clear().await.unwrap();
        test_demorepo_generic::<Neo4jGraph>(&repos).await.unwrap();
        std::env::remove_var("NEO4J_TEST_NAMESPACE");
    }
}
