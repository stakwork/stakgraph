use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::linker::{normalize_backend_path, normalize_frontend_path};
use crate::lang::{Graph, Node};
use crate::utils::get_use_lsp;
use crate::{
    lang::Lang,
    repo::{Repo, Repos},
};
use core::panic;
use shared::error::Result;
use std::str::FromStr;

pub async fn test_nextjs_generic<G: Graph>() -> Result<()> {
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

    // graph.analysis();

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
    assert_eq!(file_nodes.len(), 55, "Expected 55 File nodes");

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
    assert_eq!(directory_nodes.len(), 28, "Expected 28 Directory nodes");

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 Repository node");

    let repo = Node::new(
        NodeType::Repository,
        repository
            .first()
            .expect("Repository node not found")
            .clone(),
    );

    let app_dir = directory_nodes
        .iter()
        .find(|d| d.name == "app" && d.file.ends_with("nextjs/app"))
        .map(|n| Node::new(NodeType::Directory, n.clone()))
        .expect("Directory 'app' not found");

    let items_dir = directory_nodes
        .iter()
        .find(|d| d.name == "items" && d.file.ends_with("nextjs/app/items"))
        .map(|n| Node::new(NodeType::Directory, n.clone()))
        .expect("Directory 'items' not found");

    let items_page = file_nodes
        .iter()
        .find(|f| f.name == "page.tsx" && f.file.ends_with("nextjs/app/items/page.tsx"))
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("File 'ItemsPage.tsx' not found");

    // / -> /app -> /app/items -> /app/items/page.tsx
    assert!(graph.has_edge(&repo, &app_dir, EdgeType::Contains));
    assert!(graph.has_edge(&app_dir, &items_dir, EdgeType::Contains));
    assert!(graph.has_edge(&items_dir, &items_page, EdgeType::Contains));

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 21, "Expected 21 Endpoint nodes");

    let requests = graph.find_nodes_by_type(NodeType::Request);
    nodes += requests.len();
    assert_eq!(requests.len(), 19, "Expected 19 Request nodes");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    if use_lsp {
        // assert_eq!(functions.len(), 57, "Expected 57 Function nodes with LSP");
    } else {
        assert_eq!(
            functions.len(),
            73,
            "Expected 73 Function nodes without LSP"
        );
    }

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes += pages.len();
    assert_eq!(pages.len(), 4, "Expected Page nodes");

    let app_page = pages
        .iter()
        .find(|p| p.name == "app" && p.file.ends_with("nextjs/app/page.tsx") && p.body == "/")
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Home' not found");

    let items_page = pages
        .iter()
        .find(|p| {
            p.name == "items" && p.file.ends_with("nextjs/app/items/page.tsx") && p.body == "/items"
        })
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Items' not found");

    let person_page = pages
        .iter()
        .find(|p| {
            p.name == "person"
                && p.file.ends_with("nextjs/app/person/page.tsx")
                && p.body == "/person"
        })
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Person' not found");

    let _docs_page = pages
        .iter()
        .find(|p| {
            p.name == "docs" && p.file.ends_with("nextjs/app/docs/page.mdx") && p.body == "/docs"
        })
        .map(|n| Node::new(NodeType::Page, n.clone()))
        .expect("Page 'Docs' not found");

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
        .expect("Function 'cn' not found");

    if let Some(cn_func) = functions
        .iter()
        .find(|f| f.name == "cn" && f.file.ends_with("nextjs/lib/utils.ts"))
    {
        assert_eq!(
            cn_func.docs.as_deref(),
            Some("Merge class names conditionally\nAccepts any number of class values"),
            "Expected docs for 'cn' function not found or mismatched"
        );
    } else {
        panic!("Function 'cn' not found for docs assertion");
    }

    if let Some(card_fn) = functions
        .iter()
        .find(|f| f.name == "Card" && f.file.ends_with("nextjs/components/ui/card.tsx"))
    {
        assert_eq!(
            card_fn.docs.as_deref(),
            Some("Card component container\nProvides base styling and layout"),
            "Expected docs for 'Card' function not found or mismatched"
        );
    } else {
        panic!("Function 'Card' not found for docs assertion");
    }

    let card_func = functions
        .iter()
        .find(|f| f.name == "Card" && f.file.ends_with("nextjs/components/ui/card.tsx"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Function 'Card' not found");

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 11, "Expected 11 Variable nodes");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 18, "Expected 18 Library nodes");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;

    //TODO: Fix lsp calls edge count : locally, it says 74 but on CI it says something else
    // assert_eq!(calls, 74, "Expected 74 Calls edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 246, "Expected 246 Contains edges");

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges += handlers;
    assert_eq!(handlers, 21, "Expected 21 Handler edges");

    let tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes += tests.len();
    assert_eq!(tests.len(), 6, "Expected 6 UnitTest nodes");

    #[cfg(not(feature = "neo4j"))]
    if let Some(_currency_test) = tests.iter().find(|t| {
        t.name == "unit: currency conversion" && t.file.ends_with("test/currency.test.ts")
    }) {
        let all_convert_sats_calls: Vec<_> = graph
            .get_edges_vec()
            .into_iter()
            .filter(|e| {
                e.edge == EdgeType::Calls
                    && e.source.node_data.file.ends_with("test/currency.test.ts")
                    && e.target.node_data.name == "convertSatsToUSD"
            })
            .collect();

        let calls_currency_version = all_convert_sats_calls
            .iter()
            .any(|e| e.target.node_data.file.ends_with("lib/currency.ts"));

        let calls_helpers_version = all_convert_sats_calls
            .iter()
            .any(|e| e.target.node_data.file.ends_with("lib/helpers.ts"));

        assert!(
            calls_currency_version,
            "Currency test should call convertSatsToUSD from lib/currency.ts. Found calls: {:?}",
            all_convert_sats_calls
                .iter()
                .map(|e| format!("{} -> {}", e.source.node_data.file, e.target.node_data.file))
                .collect::<Vec<_>>()
        );

        assert!(
            !calls_helpers_version,
            "Currency test should NOT call convertSatsToUSD from lib/helpers.ts"
        );
    } else {
        panic!("Unit test 'unit: currency conversion' not found");
    }

    if let Some(test) = tests
        .iter()
        .filter(|n| n.file.ends_with("nextjs/app/test/unit.test.ts") && n.name == "unit: utils.cn")
        .next()
    {
        let test_body = format!(
            r#"describe("unit: utils.cn", () => {{
  it("merges class names", () => {{
    const result = cn("btn", "btn-primary");
    expect(result).toBe("btn btn-primary");
    console.log("cn result:", result);
  }});
}})"#
        );
        assert_eq!(test.name, "unit: utils.cn");
        assert_eq!(test.body, test_body,)
    } else {
        panic!("Unit test 'unit: utils.cn' not found");
    }

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 2, "Expected 2 Class nodes");

    let calculator_class = classes
        .iter()
        .find(|c| c.name == "Calculator")
        .expect("Calculator class not found");

    assert!(calculator_class.file.ends_with("nextjs/lib/calculator.ts"));

    if let Some(calc_test) = tests
        .iter()
        .find(|n| n.file.ends_with("nextjs/app/test/unit.class.test.ts"))
    {
        assert_eq!(calc_test.name, "unit: Calculator class");

        let calc_methods = functions
            .iter()
            .filter(|f| {
                f.file.ends_with("nextjs/lib/calculator.ts")
                    && ["add", "multiply", "subtract", "getResult"].contains(&f.name.as_str())
            })
            .collect::<Vec<_>>();

        assert_eq!(calc_methods.len(), 4, "Expected 4 Calculator methods");
    } else {
        panic!("Unit test 'unit: Calculator class' not found");
    }

    let integration_test = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes += integration_test.len();
    assert_eq!(
        integration_test.len(),
        11,
        "Expected 11 IntegrationTest nodes"
    );

    if let Some(test) = integration_test
        .iter()
        .filter(|n| {
            n.file.ends_with("nextjs/app/test/integration.test.ts")
                && n.name == "integration: /api/items"
        })
        .next()
    {
        assert_eq!(test.name, "integration: /api/items");
    } else {
        panic!("Integration test 'integration: /api/items' not found");
    }

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes += e2e_tests.len();

    assert_eq!(e2e_tests.len(), 5, "Expected 5 E2eTest nodes");

    if let Some(test) = e2e_tests
        .iter()
        .filter(|n| n.file.ends_with("nextjs/app/test/e2e.test.ts") && n.name == "e2e: user flows")
        .next()
    {
        let test_body = format!(
            r#"describe("e2e: user flows", () => {{
  it("navigates to /items and adds an item", async () => {{
    await page.goto("http://localhost:3000/items");
    await page.getByPlaceholder("Title").fill("Item 1");
    await page.getByPlaceholder("Price").fill("10");
    await page.getByRole("button", {{ name: "Add Item" }}).click();
    await expect(page.getByText("Item 1 - $10")).toBeVisible();
    console.log("E2E flow: add item");
  }});

  it("navigates to /person and manages a person by id", async () => {{
    await page.goto("http://localhost:3000/person");
    await page.getByPlaceholder("Name").fill("Alice");
    await page.getByPlaceholder("Age").fill("30");
    await page.getByPlaceholder("Email").fill("alice@example.com");
    await page.getByRole("button", {{ name: "Add Person" }}).click();
    await page.getByPlaceholder("Enter person ID (1, 2, 3...)").fill("1");
    await page.getByRole("button", {{ name: "Find" }}).click();
    await expect(page.getByText("Found: Alice")).toBeVisible();
    console.log("E2E flow: add, find and delete person");
  }});
}})"#
        );

        assert_eq!(test.name, "e2e: user flows");
        assert_eq!(test.body, test_body,)
    } else {
        panic!("E2E test 'e2e: user flowse' not found");
    }

    let import = graph.count_edges_of_type(EdgeType::Imports);
    edges += import;
    if use_lsp {
        //  assert_eq!(import, 18, "Expected 18 Imports edges with LSP");
    } else {
        assert_eq!(import, 8, "Expected 8 Imports edges without LSP");
    }

    let import_nodes = graph.find_nodes_by_type(NodeType::Import);
    nodes += import_nodes.len();
    assert_eq!(import_nodes.len(), 24, "Expected 24 Import nodes");

    let datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += datamodels.len();
    assert_eq!(datamodels.len(), 3, "Expected 3 DataModel nodes");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges += uses;
    if use_lsp {
        // assert_eq!(uses, 37, "Expected 37 Uses edges with LSP");
    } else {
        //  assert_eq!(uses, 0, "Expected 0 Uses edge without LSP");
    }

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges += nested_in;
    assert_eq!(nested_in, 5, "Expected 5 NestedIn edges");

    let operand = graph.count_edges_of_type(EdgeType::Operand);
    edges += operand;

    #[cfg(not(feature = "neo4j"))]{
        assert_eq!(
            operand, 12,
            "Expected 12  Operand edges)"
        );
    }

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

#[cfg(all(feature = "neo4j", feature = "fulltest"))]
async fn test_remote_nextjs() -> Result<()> {
    use crate::lang::graphs::Neo4jGraph;
    let repo_url = "https://github.com/clerk/clerk-nextjs-demo-pages-router";
    let use_lsp = get_use_lsp();
    let repos = Repo::new_clone_multi_detect(
        repo_url,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
        None,
        Some(use_lsp),
    )
    .await?;

    let graph = Neo4jGraph::default();
    graph.clear().await?;
    let graph = repos.build_graphs_inner::<Neo4jGraph>().await?;
    // graph.analysis();

    let mut nodes = 0;
    let mut edges = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "react",
        "Language node name should be 'react'"
    );

    let repository = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repository.len();
    assert_eq!(repository.len(), 1, "Expected 1 Repository node");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directories.len();
    assert_eq!(directories.len(), 9, "Expected 9 Directory nodes");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    nodes += file_nodes.len();
    assert_eq!(file_nodes.len(), 28, "Expected 28 File nodes");

    let pages = graph.find_nodes_by_type(NodeType::Page);
    nodes += pages.len();
    assert_eq!(pages.len(), 4, "Expected 4 Page nodes (Pages Router)");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    if use_lsp {
        assert_eq!(
            functions.len(),
            29,
            "Expected 29 Function nodes (Pages Router)"
        );
    } else {
        assert_eq!(
            functions.len(),
            25,
            "Expected 25 Function nodes (Pages Router)"
        );
    }

    let imports = graph.count_edges_of_type(EdgeType::Imports);
    edges += imports;
    if use_lsp {
        assert_eq!(imports, 30, "Expected 30 Imports edges");
    } else {
        assert_eq!(imports, 10, "Expected 10 Imports edges");
    }

    let import_nodes = graph.find_nodes_by_type(NodeType::Import);
    nodes += import_nodes.len();
    assert_eq!(import_nodes.len(), 18, "Expected 18 Import nodes");

    let library = graph.find_nodes_by_type(NodeType::Library);
    nodes += library.len();
    assert_eq!(library.len(), 12, "Expected 12 Library nodes");

    let next_lib = library
        .iter()
        .find(|l| {
            l.name == "next"
                && l.file
                    .ends_with("clerk/clerk-nextjs-demo-pages-router/package.json")
        })
        .map(|n| Node::new(NodeType::Library, n.clone()))
        .expect("Next.js library not found");

    let pkg_file = file_nodes
        .iter()
        .find(|f| {
            f.name == "package.json"
                && f.file
                    .ends_with("clerk/clerk-nextjs-demo-pages-router/package.json")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("package.json file not found");

    assert!(
        graph.has_edge(&pkg_file, &next_lib, EdgeType::Contains),
        "package.json should contain Next.js library"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 4, "Expected 4 Variable nodes");

    let datamodels = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += datamodels.len();
    assert_eq!(datamodels.len(), 4, "Expected 4 DataModel nodes");

    let session_file = file_nodes
        .iter()
        .find(|f| {
            f.name == "SessionDetails.tsx"
                && f.file.ends_with("src/pages/dashboard/SessionDetails.tsx")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("Session file not found");
    let session_window = datamodels
        .iter()
        .find(|d| d.name == "Window" && d.file.ends_with("src/pages/dashboard/SessionDetails.tsx"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Window data model for session not found");

    assert!(
        graph.has_edge(&session_file, &session_window, EdgeType::Contains),
        "SessionDetails should contain Window data model"
    );

    let user_details_window = datamodels
        .iter()
        .find(|d| d.name == "Window" && d.file.ends_with("src/pages/dashboard/UserDetails.tsx"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Window data model for user details not found");

    let user_details_file = file_nodes
        .iter()
        .find(|f| {
            f.name == "UserDetails.tsx" && f.file.ends_with("src/pages/dashboard/UserDetails.tsx")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("UserDetails file not found");

    assert!(
        graph.has_edge(&user_details_file, &user_details_window, EdgeType::Contains),
        "UserDetails should contain Window data model"
    );

    let org_details_window = datamodels
        .iter()
        .find(|d| d.name == "Window" && d.file.ends_with("src/pages/dashboard/OrgDetails.tsx"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Window data model for org details not found");

    let org_details_file = file_nodes
        .iter()
        .find(|f| {
            f.name == "OrgDetails.tsx" && f.file.ends_with("src/pages/dashboard/OrgDetails.tsx")
        })
        .map(|n| Node::new(NodeType::File, n.clone()))
        .expect("OrgDetails file not found");

    assert!(
        graph.has_edge(&org_details_file, &org_details_window, EdgeType::Contains),
        "OrgDetails should contain Window data model"
    );

    let data = datamodels
        .iter()
        .find(|d| d.name == "Data" && d.file.ends_with("src/pages/api/hello.ts"))
        .map(|n| Node::new(NodeType::DataModel, n.clone()))
        .expect("Data data model not found");

    let handler = functions
        .iter()
        .find(|f| f.name == "handler" && f.file.ends_with("src/pages/api/hello.ts"))
        .map(|n| Node::new(NodeType::Function, n.clone()))
        .expect("Handler function not found");

    assert!(
        graph.has_edge(&handler, &data, EdgeType::Contains),
        "SessionDetails should contain Window data model"
    );

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls;
    // assert_eq!(calls, 31, "Expected 31 Calls edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 102, "Expected 102 Contains edges");

    let renders = graph.count_edges_of_type(EdgeType::Renders);
    edges += renders;
    assert_eq!(renders, 4, "Expected 4 Renders edges");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges += uses;
    if use_lsp {
        // assert_eq!(uses, 4, "Expected 4 Uses edges with LSP");
    } else {
        // assert_eq!(uses, 0, "Expected 0 Uses edge without LSP");
    }

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges += nested_in;
    // assert_eq!(nested_in, 4, "Expected 4 NestedIn edges"); to be udpated

    let sign_in_page = pages
        .iter()
        .find(|p| {
            p.name == "sign-in" && p.file.ends_with("src/pages/sign-in.tsx") && p.body == "/sign-in"
        })
        .expect("sign-in page not found");
    let sign_up_page = pages
        .iter()
        .find(|p| {
            p.name == "sign-up" && p.file.ends_with("src/pages/sign-up.tsx") && p.body == "/sign-up"
        })
        .expect("sign-up page not found");
    let dashboard_page = pages
        .iter()
        .find(|p| {
            p.name == "dashboard"
                && p.file.ends_with("src/pages/dashboard/index.tsx")
                && p.body == "/dashboard"
        })
        .expect("dashboard page not found");
    let index_page = pages
        .iter()
        .find(|p| p.name == "index" && p.file.ends_with("src/pages/index.tsx") && p.body == "/")
        .expect("index page not found");

    let sign_in_component = functions
        .iter()
        .find(|f| f.name == "SignInPage" && f.file.ends_with("src/pages/sign-in.tsx"))
        .expect("SignInPage component not found");
    let sign_up_component = functions
        .iter()
        .find(|f| f.name == "SignUpPage" && f.file.ends_with("src/pages/sign-up.tsx"))
        .expect("SignUpPage component not found");
    let dashboard_component = functions
        .iter()
        .find(|f| f.name == "DashboardPage" && f.file.ends_with("src/pages/dashboard/index.tsx"))
        .expect("DashboardPage component not found");
    let home_component = functions
        .iter()
        .find(|f| f.name == "Home" && f.file.ends_with("src/pages/index.tsx"))
        .expect("Home component not found");

    assert!(
        graph.has_edge(
            &Node::new(NodeType::Page, sign_in_page.clone()),
            &Node::new(NodeType::Function, sign_in_component.clone()),
            EdgeType::Renders
        ),
        "sign-in page should render SignInPage"
    );
    assert!(
        graph.has_edge(
            &Node::new(NodeType::Page, sign_up_page.clone()),
            &Node::new(NodeType::Function, sign_up_component.clone()),
            EdgeType::Renders
        ),
        "sign-up page should render SignUpPage"
    );
    assert!(
        graph.has_edge(
            &Node::new(NodeType::Page, dashboard_page.clone()),
            &Node::new(NodeType::Function, dashboard_component.clone()),
            EdgeType::Renders
        ),
        "dashboard page should render DashboardPage"
    );
    assert!(
        graph.has_edge(
            &Node::new(NodeType::Page, index_page.clone()),
            &Node::new(NodeType::Function, home_component.clone()),
            EdgeType::Renders
        ),
        "index page should render Home"
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
        #[cfg(feature = "fulltest")]
        test_remote_nextjs().await.unwrap();
        // cargo test test_nextjs --features neo4j --features fulltest -- --nocapture

        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_nextjs_generic::<Neo4jGraph>().await.unwrap();
    }
}
