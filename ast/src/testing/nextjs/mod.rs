use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::linker::{normalize_backend_path, normalize_frontend_path};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{
    lang::Lang,
    repo::{Repo, Repos},
};
use anyhow::Result;
use std::str::FromStr;

pub async fn test_nextjs_generic<G: Graph>() -> Result<(), anyhow::Error> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
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
    assert_eq!(
        language_nodes[0].name, "react",
        "Language node name should be 'tsx'"
    );

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes += file_nodes.len();
    assert_eq!(file_nodes.len(), 20, "Expected 20 File nodes");

    let card_file = file_nodes
        .iter()
        .find(|f| f.name == "card.tsx" && f.file.ends_with("nextjs/components/ui/card.tsx"))
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'Card.tsx' not found");

    let items_page_file = file_nodes
        .iter()
        .find(|f| f.name == "page.tsx" && f.file.ends_with("nextjs/app/items/page.tsx"))
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'ItemsPage.tsx' not found");

    let person_file = file_nodes
        .iter()
        .find(|f| f.name == "page.tsx" && f.file.ends_with("nextjs/app/person/page.tsx"))
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'Person.ts' not found");

    let directory_nodes = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directory_nodes.len();
    assert_eq!(directory_nodes.len(), 10, "Expected 10 Directory nodes");

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 Repository node");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 6, "Expected 6 Endpoint nodes");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes += requests.len();
    assert_eq!(requests.len(), 9, "Expected 9 Request nodes");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();

    assert_eq!(
        functions.len(),
        28,
        "Expected 28 Function nodes without LSP"
    );

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes += pages.len();
    assert_eq!(pages.len(), 3, "Expected 3 Page nodes");

    let app_page = pages
        .iter()
        .find(|p| p.name == "app" && p.file.ends_with("nextjs/app/page.tsx"))
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Home' not found");

    let items_page = pages
        .iter()
        .find(|p| p.name == "items" && p.file.ends_with("nextjs/app/items/page.tsx"))
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Items' not found");

    let person_page = pages
        .iter()
        .find(|p| p.name == "person" && p.file.ends_with("nextjs/app/person/page.tsx"))
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Person' not found");

    let home_component = functions
        .iter()
        .find(|f| f.name == "Home" && f.file.ends_with("nextjs/app/page.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Home' not found");

    let items_component = functions
        .iter()
        .find(|f| f.name == "Items" && f.file.ends_with("nextjs/app/items/page.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Items' not found");
    let person_component = functions
        .iter()
        .find(|f| f.name == "Person" && f.file.ends_with("nextjs/app/person/page.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Person' not found");

    assert!(
        graph.has_edge(&app_page, &home_component, EdgeType::Renders),
        "Home page should render Home component"
    );
    assert!(
        graph.has_edge(&items_page, &items_component, EdgeType::Renders),
        "Items page should render Items component"
    );
    assert!(
        graph.has_edge(&person_page, &person_component, EdgeType::Renders),
        "Person page should render Person component"
    );

    let cn = functions
        .iter()
        .find(|f| f.name == "cn" && f.file.ends_with("nextjs/lib/utils.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Card' not found");

    let card_func = functions
        .iter()
        .find(|f| f.name == "Card" && f.file.ends_with("nextjs/components/ui/card.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Card' not found");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 8, "Expected 8 Variable nodes");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 18, "Expected 18 Library nodes");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;
    assert_eq!(calls, 47, "Expected 47 Calls edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 112, "Expected 112 Contains edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges += handlers;
    assert_eq!(handlers, 6, "Expected 6 Handler edges");

    let import = graph.count_edges_of_type(EdgeType::Imports);
    edges += import;
    if use_lsp {
        assert_eq!(import, 15, "Expected 15 Imports edges with LSP");
    } else {
        assert_eq!(import, 0, "Expected 0 Imports edge without LSP");
    }

    let import_nodes = graph.find_nodes_by_type(NodeType::Import);
    nodes += import_nodes.len();
    assert_eq!(import_nodes.len(), 13, "Expected 13 Import nodes");

    let datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += datamodels.len();
    assert_eq!(datamodels.len(), 3, "Expected 3 DataModel nodes");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges += uses;
    assert_eq!(uses, 0, "Expected 0 Uses edge without LSP");

    let renders = graph.count_edges_of_type(EdgeType::Renders);
    edges += renders;
    assert_eq!(renders, 3, "Expected 3 Renders edges");

    let items_page_func = functions
        .iter()
        .find(|f| f.name == "Items")
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Items' not found");

    let get_items_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/api/items" && e.meta.get("verb") == Some(&"GET".to_string()))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/items endpoint not found");

    let post_items_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/api/items" && e.meta.get("verb") == Some(&"POST".to_string()))
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("POST /api/items endpoint not found");

    let get_items_handler_func = functions
        .iter()
        .find(|f| f.name == "GET" && f.file.ends_with("app/api/items/route.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("GET handler function for items not found");

    let post_items_handler_func = functions
        .iter()
        .find(|f| f.name == "POST" && f.file.ends_with("app/api/items/route.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("POST handler function for items not found");

    let get_items_request = requests
        .iter()
        .find(|r| {
            r.name == "/api/items"
                && r.meta.get("verb") == Some(&"GET".to_string())
                && r.file.ends_with("app/items/page.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("GET request to /api/items not found");

    let post_items_request = requests
        .iter()
        .find(|r| {
            r.name == "/api/items"
                && r.meta.get("verb") == Some(&"POST".to_string())
                && r.file.ends_with("app/items/page.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("POST request to /api/items not found");
    let person_page_func = functions
        .iter()
        .find(|f| f.name == "Person")
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Person' not found");
    let expected_endpoint_path = normalize_backend_path("/api/person/[id]").unwrap();

    let get_person_endpoint = endpoints
        .iter()
        .find(|e| {
            let normalized_endpoint = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
            normalized_endpoint == expected_endpoint_path
                && e.meta.get("verb") == Some(&"GET".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("GET /api/person/[id] endpoint not found");

    let get_person_handler_func = functions
        .iter()
        .find(|f| f.name == "GET" && f.file.ends_with("app/api/person/[id]/route.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("GET handler function for dynamic person route not found");

    let delete_person_handler_func = functions
        .iter()
        .find(|f| f.name == "DELETE" && f.file.ends_with("app/api/person/[id]/route.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("DELETE handler function for dynamic person route not found");

    let expected_request_path = normalize_frontend_path("/api/person/${id}").unwrap();

    let get_person_request = requests
        .iter()
        .find(|r| {
            let normalized_request = normalize_frontend_path(&r.name).unwrap_or(r.name.clone());
            normalized_request == expected_request_path
                && r.meta.get("verb") == Some(&"GET".to_string())
                && r.file.ends_with("app/person/page.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("GET request to dynamic person route not found");

    let delete_person_request = requests
        .iter()
        .find(|r| {
            let normalized_request = normalize_frontend_path(&r.name).unwrap_or(r.name.clone());
            normalized_request == expected_request_path
                && r.meta.get("verb") == Some(&"DELETE".to_string())
                && r.file.ends_with("app/person/page.tsx")
        })
        .map(|n| Node::new(NodeType::Request, n.clone()))
        .expect("DELETE request to dynamic person route not found");

    let delete_person_endpoint = endpoints
        .iter()
        .find(|e| {
            let normalized_endpoint = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
            normalized_endpoint == expected_endpoint_path
                && e.meta.get("verb") == Some(&"DELETE".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n.clone()))
        .expect("DELETE /api/person/[id] endpoint not found");

    assert!(
        graph.has_edge(
            &get_items_endpoint,
            &get_items_handler_func,
            EdgeType::Handler
        ),
        "Expected GET /api/items endpoint to be handled by GET function"
    );
    assert!(
        graph.has_edge(
            &post_items_endpoint,
            &post_items_handler_func,
            EdgeType::Handler
        ),
        "Expected POST /api/items endpoint to be handled by POST function"
    );

    assert!(
        graph.has_edge(&items_page_func, &get_items_request, EdgeType::Calls),
        "Expected Items to call the GET /api/items request"
    );
    assert!(
        graph.has_edge(&items_page_func, &post_items_request, EdgeType::Calls),
        "Expected ItemsPage to call the POST /api/items request"
    );

    assert!(
        graph.has_edge(&get_items_request, &get_items_endpoint, EdgeType::Calls),
        "Expected GET request to call the GET /api/items endpoint"
    );
    assert!(
        graph.has_edge(&post_items_request, &post_items_endpoint, EdgeType::Calls),
        "Expected POST request to call the POST /api/items endpoint"
    );
    assert!(
        graph.has_edge(
            &get_person_endpoint,
            &get_person_handler_func,
            EdgeType::Handler
        ),
        "Expected GET dynamic endpoint to be handled by its GET function"
    );
    assert!(
        graph.has_edge(
            &delete_person_endpoint,
            &delete_person_handler_func,
            EdgeType::Handler
        ),
        "Expected DELETE dynamic endpoint to be handled by its DELETE function"
    );

    assert!(
        graph.has_edge(&person_page_func, &get_person_request, EdgeType::Calls),
        "Expected Person to call the dynamic GET person request"
    );
    assert!(
        graph.has_edge(&person_page_func, &delete_person_request, EdgeType::Calls),
        "Expected Person to call the dynamic DELETE person request"
    );

    assert!(
        graph.has_edge(&get_person_request, &get_person_endpoint, EdgeType::Calls),
        "Expected dynamic GET request to call the dynamic GET endpoint"
    );
    assert!(
        graph.has_edge(
            &delete_person_request,
            &delete_person_endpoint,
            EdgeType::Calls
        ),
        "Expected dynamic DELETE request to call the dynamic DELETE endpoint"
    );

    assert!(
        graph.has_edge(&card_file, &card_func, EdgeType::Contains),
        "Expected Card file to call the Card function"
    );

    assert!(
        graph.has_edge(&card_func, &cn, EdgeType::Calls),
        "Expected Card function to call the cn function"
    );

    assert!(
        graph.has_edge(&items_page_file, &items_page_func, EdgeType::Contains),
        "Expected ItemsPage file to contain the ItemsPage function"
    );

    assert!(
        graph.has_edge(&person_file, &person_page_func, EdgeType::Contains),
        "Expected Person file to contain the PersonPage function"
    );

    if use_lsp {
        assert!(
            graph.has_edge(&person_file, &card_func, EdgeType::Imports),
            "Expected Person file to import Card function"
        );
        assert!(
            graph.has_edge(&items_page_file, &card_func, EdgeType::Imports),
            "Expected ItemsPage file to import Card function"
        );
    }
    assert!(
        graph.has_edge(&person_page_func, &card_func, EdgeType::Calls),
        "Expected PersonPage function to call Card function"
    );
    assert!(
        graph.has_edge(&items_page_func, &card_func, EdgeType::Calls),
        "Expected ItemsPage function to call Card function"
    );

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(
        num_nodes, nodes as u32,
        "Nodes mismatch: expected {num_nodes} nodes found {nodes}"
    );
    assert_eq!(
        num_edges, edges as u32,
        "Edges mismatch: expected {num_edges} edges found {edges}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nextjs() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_nextjs_generic::<ArrayGraph>().await.unwrap();
    test_nextjs_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_nextjs_generic::<Neo4jGraph>().await.unwrap();
    }
}
