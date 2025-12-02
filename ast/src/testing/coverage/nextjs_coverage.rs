// use crate::lang::graphs::{EdgeType, Graph, NodeType};
// use crate::lang::linker::normalize_backend_path;
// use crate::lang::Lang;
// use crate::repo::{Repo, Repos};
// use crate::utils::get_use_lsp;
// use shared::error::Result;
// use std::str::FromStr;

// #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
// async fn test_integration_coverage() -> Result<()> {
//     let use_lsp = get_use_lsp();

//     println!("Testing Next.js integration coverage");
//     let repo = Repo::new(
//         "src/testing/nextjs",
//         Lang::from_str("tsx").unwrap(),
//         use_lsp,
//         Vec::new(),
//         Vec::new(),
//     )
//     .unwrap();

//     let repos = Repos(vec![repo]);
//     let btree_graph = repos.build_graphs_btree().await?;

//     btree_graph.analysis();

//     let endpoints = btree_graph.find_nodes_by_type(NodeType::Endpoint);
//     assert_eq!(endpoints.len(), 21, "Expected exactly 21 endpoints");

//     let integration_tests = btree_graph.find_nodes_by_type(NodeType::IntegrationTest);
//     assert_eq!(
//         integration_tests.len(),
//         11,
//         "Expected exactly 11 integration test describe blocks"
//     );

//     let test_edges = btree_graph.find_nodes_with_edge_type(
//         NodeType::IntegrationTest,
//         NodeType::Endpoint,
//         EdgeType::Calls,
//     );

//     let unique_tested_endpoints: std::collections::HashSet<String> = test_edges
//         .iter()
//         .map(|(_, target)| format!("{}|{}|{}", target.name, target.file, target.start))
//         .collect();

//     assert_eq!(
//         unique_tested_endpoints.len(),
//         10,
//         "Expected exactly 10 unique tested endpoints directly called by tests"
//     );

//     // Check endpoints with indirect_test metadata (linked via helper functions)
//     let indirect_tested: Vec<_> = endpoints
//         .iter()
//         .filter(|e| e.meta.contains_key("indirect_test"))
//         .collect();

//     assert_eq!(
//         indirect_tested.len(),
//         4,
//         "Expected exactly 4 endpoints with indirect_test metadata (2 /api/items + 2 /api/products from Pattern 1)"
//     );

//     // Verify the indirect endpoints
//     let expected_indirect = vec![
//         ("/api/items", "POST", "createItem"),
//         ("/api/items", "GET", "fetchItems"),
//         ("/api/products", "POST", "createProduct"),
//         ("/api/products", "GET", "listProducts"),
//     ];

//     for (path, verb, helper_name) in &expected_indirect {
//         let matching: Vec<_> = indirect_tested
//             .iter()
//             .filter(|e| {
//                 e.name.contains(path)
//                     && e.meta.get("verb") == Some(&verb.to_string())
//                     && !e.name.contains("[")
//             })
//             .collect();

//         assert_eq!(
//             matching.len(),
//             1,
//             "Expected exactly 1 indirect endpoint for {} {}",
//             verb,
//             path
//         );

//         let endpoint = matching[0];
//         assert!(
//             endpoint.meta.contains_key("test_helper"),
//             "Endpoint {} {} should have test_helper metadata",
//             verb,
//             path
//         );

//         let helper = endpoint.meta.get("test_helper").unwrap();
//         assert!(
//             helper.contains(helper_name),
//             "Endpoint {} {} should have helper '{}', got '{}'",
//             verb,
//             path,
//             helper_name,
//             helper
//         );
//     }

//     let tested_endpoints = vec![
//         ("/api/items", "GET"),
//         ("/api/items", "POST"),
//         ("/api/person", "GET"),
//         ("/api/person", "POST"),
//         ("/api/person/:param", "GET"),
//         ("/api/person/:param", "DELETE"),
//         ("/api/orders", "PUT"),
//     ];

//     for (expected_path, expected_verb) in &tested_endpoints {
//         let matching_endpoints: Vec<_> = endpoints
//             .iter()
//             .filter(|e| {
//                 let normalized = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
//                 normalized == *expected_path
//                     && e.meta.get("verb") == Some(&expected_verb.to_string())
//             })
//             .collect();

//         assert_eq!(
//             matching_endpoints.len(),
//             1,
//             "Expected exactly 1 endpoint matching {} {}, found {}",
//             expected_verb,
//             expected_path,
//             matching_endpoints.len()
//         );

//         let endpoint = matching_endpoints[0];

//         let has_test_edge = test_edges.iter().any(|(_, target)| {
//             target.name == endpoint.name
//                 && target.file == endpoint.file
//                 && target.start == endpoint.start
//         });

//         assert_eq!(
//             has_test_edge, true,
//             "Endpoint {} {} should have integration test edge",
//             expected_verb, expected_path
//         );
//     }

//     let untested_endpoints = vec![("/api/users", "POST"), ("/api/products/:param", "GET")];

//     for (expected_path, expected_verb) in &untested_endpoints {
//         let matching_endpoints: Vec<_> = endpoints
//             .iter()
//             .filter(|e| {
//                 let normalized = normalize_backend_path(&e.name).unwrap_or(e.name.clone());
//                 normalized == *expected_path
//                     && e.meta.get("verb") == Some(&expected_verb.to_string())
//             })
//             .collect();

//         assert_eq!(
//             matching_endpoints.len(),
//             1,
//             "Expected exactly 1 endpoint matching {} {}, found {}",
//             expected_verb,
//             expected_path,
//             matching_endpoints.len()
//         );

//         let endpoint = matching_endpoints[0];

//         let has_test_edge = test_edges.iter().any(|(_, target)| {
//             target.name == endpoint.name
//                 && target.file == endpoint.file
//                 && target.start == endpoint.start
//         });

//         assert_eq!(
//             has_test_edge, false,
//             "Endpoint {} {} should NOT have integration test edge",
//             expected_verb, expected_path
//         );
//     }

//     #[cfg(feature = "neo4j")]
//     {
//         use crate::lang::graphs::graph_ops::GraphOps;

//         // Phase 3: Upload to Neo4j
//         let mut graph_ops = GraphOps::new();
//         graph_ops.connect().await?;
//         graph_ops.graph.clear().await?;
//         graph_ops
//             .upload_btreemap_to_neo4j(&btree_graph, None)
//             .await?;

//         // Phase 4: Query tested endpoints using query_nodes_with_count
//         let (tested_count, tested_results) = graph_ops
//             .query_nodes_with_count(
//                 NodeType::Endpoint,
//                 0,
//                 100,
//                 false,
//                 Some("tested"),
//                 false,
//                 false,
//                 None,
//                 None,
//                 None,
//             )
//             .await?;

//         assert_eq!(
//             tested_count, 12,
//             "Neo4j should report exactly 12 tested endpoints"
//         );

//         assert_eq!(
//             tested_results.len(),
//             12,
//             "Neo4j should return exactly 12 tested endpoint results"
//         );

//         let (untested_count, untested_results) = graph_ops
//             .query_nodes_with_count(
//                 NodeType::Endpoint,
//                 0,
//                 100,
//                 false,
//                 Some("untested"),
//                 false,
//                 false,
//                 None,
//                 None,
//                 None,
//             )
//             .await?;

//         assert_eq!(
//             untested_count, 9,
//             "Neo4j should report exactly 9 untested endpoints"
//         );

//         assert_eq!(
//             untested_results.len(),
//             9,
//             "Neo4j should return exactly 9 untested endpoint results"
//         );
//     }

//     Ok(())
// }
