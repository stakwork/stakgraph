#[cfg(all(feature = "neo4j", feature = "fulltest"))]
mod test_endpoints {
    use ast::lang::graphs::graph_ops::GraphOps;
    use ast::lang::{Graph, Lang, NodeType};
    use ast::repo::{Repo, Repos};
    use std::sync::Mutex;
    use test_log::test;
    use tracing::info;

    static SETUP_LOCK: Mutex<bool> = Mutex::new(false);

    async fn setup_nextjs_graph() -> GraphOps {
        let mut setup_complete = SETUP_LOCK.lock().unwrap();

        if !*setup_complete {
            let mut ops = GraphOps::new();
            ops.connect().await.unwrap();
            ops.clear().await.unwrap();

            let use_lsp = false;
            let repo = Repo::new(
                "../ast/src/testing/nextjs",
                Lang::from_language(lsp::Language::React),
                use_lsp,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();

            let repos = Repos(vec![repo]);
            let btree_graph = repos
                .build_graphs_inner::<ast::lang::graphs::BTreeMapGraph>()
                .await
                .unwrap();
            btree_graph.analysis();

            let (nodes, edges) = ops
                .upload_btreemap_to_neo4j(&btree_graph, None)
                .await
                .unwrap();

            info!("Graph built and uploaded: {} nodes, {} edges", nodes, edges);
            
            *setup_complete = true;
        }

        drop(setup_complete);

        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await.unwrap();
        graph_ops
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_endpoint_basic_structure() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        assert!(coverage.unit_tests.is_some());
        assert!(coverage.integration_tests.is_some());
        assert!(coverage.e2e_tests.is_some());

        let unit = coverage.unit_tests.unwrap();
        assert_eq!(unit.total, 1);
        assert_eq!(unit.total_tests, 4);
        assert_eq!(unit.covered, 1);
        assert_eq!(unit.percent, 100.0);

        let integration = coverage.integration_tests.unwrap();
        assert_eq!(integration.total, 6);
        assert_eq!(integration.total_tests, 4);
        assert_eq!(integration.covered, 6);
        assert_eq!(integration.percent, 100.0);

        let e2e = coverage.e2e_tests.unwrap();
        assert_eq!(e2e.total, 4);
        assert_eq!(e2e.total_tests, 5);
        assert_eq!(e2e.covered, 3);
        assert_eq!(e2e.percent, 75.0);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_endpoint_test_counts() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        let unit = coverage.unit_tests.unwrap();
        assert_eq!(unit.total_tests, 4);
        assert_eq!(unit.covered, 1);
        assert_eq!(unit.total, 1);

        let integration = coverage.integration_tests.unwrap();
        assert_eq!(integration.total_tests, 4);
        assert_eq!(integration.covered, 6);
        assert_eq!(integration.total, 6);

        let e2e = coverage.e2e_tests.unwrap();
        assert_eq!(e2e.total_tests, 5);
        assert_eq!(e2e.covered, 3);
        assert_eq!(e2e.total, 4);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_percentage_calculation() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        let unit = coverage.unit_tests.unwrap();
        assert_eq!(unit.percent, 100.0);
        assert_eq!(
            unit.percent,
            (unit.covered as f64 / unit.total as f64) * 100.0
        );

        let integration = coverage.integration_tests.unwrap();
        assert_eq!(integration.percent, 100.0);
        assert_eq!(
            integration.percent,
            (integration.covered as f64 / integration.total as f64) * 100.0
        );

        let e2e = coverage.e2e_tests.unwrap();
        assert_eq!(e2e.percent, 75.0);
        assert_eq!(e2e.percent, (e2e.covered as f64 / e2e.total as f64) * 100.0);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_functions_all() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        info!(
            "Functions - total_count={}, results.len()={}",
            total_count,
            results.len()
        );

        assert_eq!(total_count, 1);
        assert_eq!(results.len(), 1);

        let cn_found = results
            .iter()
            .any(|(node, ..)| node.name == "cn" && node.file.contains("lib/utils.ts"));
        assert!(cn_found, "Should find 'cn' function from lib/utils.ts");
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_functions_tested() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_tested, tested_results) = graph_ops
            .query_nodes_with_count(
                NodeType::Function,
                0,
                100,
                false,
                Some("tested"),
                false,
                false,
            )
            .await
            .unwrap();

        for (_, _, covered, test_count, _, _, _) in &tested_results {
            assert_eq!(total_tested, 1);
            assert_eq!(tested_results.len(), 1);
            assert!(*covered);
            assert!(*test_count > 0);
        }
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_functions_untested() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_untested, untested_results) = graph_ops
            .query_nodes_with_count(
                NodeType::Function,
                0,
                100,
                false,
                Some("untested"),
                false,
                false,
            )
            .await
            .unwrap();

        for (_, _, covered, _, _, _, _) in &untested_results {
            assert!(
                !*covered,
                "All functions in 'untested' filter should not be covered"
            );

            assert_eq!(total_untested, 0);
            assert_eq!(untested_results.len(), 0);
        }
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_endpoints_all() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_endpoints, endpoint_results) = graph_ops
            .query_nodes_with_count(NodeType::Endpoint, 0, 100, false, None, false, false)
            .await
            .unwrap();

        assert_eq!(total_endpoints, 6);

        let get_items = endpoint_results.iter().any(|(node, ..)| {
            node.name.contains("/api/items") && node.meta.get("verb") == Some(&"GET".to_string())
        });
        assert!(get_items);

        let post_items = endpoint_results.iter().any(|(node, ..)| {
            node.name.contains("/api/items") && node.meta.get("verb") == Some(&"POST".to_string())
        });
        assert!(post_items);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_pagination() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        assert_eq!(total_count, 1);
        assert_eq!(results.len(), 1);

        let limit = 5;
        let (count1, page1) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, limit, false, None, false, false)
            .await
            .unwrap();

        assert_eq!(count1, 1);
        assert_eq!(page1.len(), 1);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_sort_by_test_count() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (_, sorted_results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 10, true, None, false, false)
            .await
            .unwrap();

        for i in 1..sorted_results.len() {
            let prev_test_count = sorted_results[i - 1].3;
            let curr_test_count = sorted_results[i].3;
            assert!(prev_test_count >= curr_test_count);
        }
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_coverage_filter_validation() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_all, _) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, Some("all"), false, false)
            .await
            .unwrap();

        let (total_no_filter, _) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        assert_eq!(total_all, 1);
        assert_eq!(total_no_filter, 1);
        assert_eq!(total_all, total_no_filter);

        let (total_tested, _) = graph_ops
            .query_nodes_with_count(
                NodeType::Function,
                0,
                100,
                false,
                Some("tested"),
                false,
                false,
            )
            .await
            .unwrap();

        let (total_untested, _) = graph_ops
            .query_nodes_with_count(
                NodeType::Function,
                0,
                100,
                false,
                Some("untested"),
                false,
                false,
            )
            .await
            .unwrap();

        assert_eq!(total_tested, 1);
        assert_eq!(total_untested, 0);
        assert_eq!(total_tested + total_untested, total_all);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_endpoints_with_coverage() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_endpoints, endpoint_results) = graph_ops
            .query_nodes_with_count(NodeType::Endpoint, 0, 100, true, None, false, false)
            .await
            .unwrap();

        assert_eq!(total_endpoints, 6);
        assert_eq!(endpoint_results.len(), 6);

        let tested_endpoints = endpoint_results
            .iter()
            .filter(|(_, _, covered, _, _, _, _)| *covered)
            .count();

        assert_eq!(tested_endpoints, 6);

        let get_items_tested =
            endpoint_results
                .iter()
                .any(|(node, _, covered, test_count, _, _, _)| {
                    node.name.contains("/api/items")
                        && node.meta.get("verb") == Some(&"GET".to_string())
                        && *covered
                        && *test_count > 0
                });

        assert!(get_items_tested);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_body_and_line_count_fields() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        info!(
            "Body/Line Count Test - total_count={}, results.len()={}",
            total_count,
            results.len()
        );

        assert_eq!(total_count, 1);
        assert_eq!(results.len(), 1);

        for (node, _, _, _, _, body_length, line_count) in &results {
            info!(
                "Function '{}' - body_length={:?}, line_count={:?}",
                node.name, body_length, line_count
            );
            
            assert_eq!(*body_length, Some(78));
            assert_eq!(*line_count, Some(2));
        }
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_sort_by_body_length() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, sorted_results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, true, false)
            .await
            .unwrap();

        info!(
            "Sort by body_length - total_count={}, results.len()={}",
            total_count,
            sorted_results.len()
        );

        for (node, _, _, _, _, body_length, line_count) in &sorted_results {
            info!(
                "Function '{}' - body_length={:?}, line_count={:?}",
                node.name, body_length, line_count
            );
        }

        assert_eq!(total_count, 1);
        assert_eq!(sorted_results.len(), 1);
        assert_eq!(sorted_results[0].5, Some(78));
        assert_eq!(sorted_results[0].6, Some(2));
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_sort_by_line_count() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, sorted_results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, true)
            .await
            .unwrap();

        info!(
            "Sort by line_count - total_count={}, results.len()={}",
            total_count,
            sorted_results.len()
        );

        for (node, _, _, _, _, body_length, line_count) in &sorted_results {
            info!(
                "Function '{}' - body_length={:?}, line_count={:?}",
                node.name, body_length, line_count
            );
        }

        assert_eq!(total_count, 1);
        assert_eq!(sorted_results.len(), 1);
        assert_eq!(sorted_results[0].5, Some(78));
        assert_eq!(sorted_results[0].6, Some(2));
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_edge_cases() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (count_zero_limit, results_zero_limit) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 0, false, None, false, false)
            .await
            .unwrap();

        info!(
            "Zero limit - total_count={}, results.len()={}",
            count_zero_limit,
            results_zero_limit.len()
        );

        assert_eq!(count_zero_limit, 1);
        assert_eq!(results_zero_limit.len(), 0);

        let (count_large_offset, results_large_offset) = graph_ops
            .query_nodes_with_count(NodeType::Function, 1000, 10, false, None, false, false)
            .await
            .unwrap();

        info!(
            "Large offset - total_count={}, results.len()={}",
            count_large_offset,
            results_large_offset.len()
        );

        assert_eq!(count_large_offset, 1);
        assert_eq!(results_large_offset.len(), 0);
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_multiple_node_types() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (func_count, _) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        let (endpoint_count, _) = graph_ops
            .query_nodes_with_count(NodeType::Endpoint, 0, 100, false, None, false, false)
            .await
            .unwrap();

        let (page_count, _) = graph_ops
            .query_nodes_with_count(NodeType::Page, 0, 100, false, None, false, false)
            .await
            .unwrap();

        let (unit_test_count, _) = graph_ops
            .query_nodes_with_count(NodeType::UnitTest, 0, 100, false, None, false, false)
            .await
            .unwrap();

        let (integration_test_count, _) = graph_ops
            .query_nodes_with_count(NodeType::IntegrationTest, 0, 100, false, None, false, false)
            .await
            .unwrap();

        let (e2e_test_count, _) = graph_ops
            .query_nodes_with_count(NodeType::E2eTest, 0, 100, false, None, false, false)
            .await
            .unwrap();

        info!("Function count: {}", func_count);
        info!("Endpoint count: {}", endpoint_count);
        info!("Page count: {}", page_count);
        info!("UnitTest count: {}", unit_test_count);
        info!("IntegrationTest count: {}", integration_test_count);
        info!("E2eTest count: {}", e2e_test_count);

        assert_eq!(func_count, 1);
        assert_eq!(endpoint_count, 6);
        assert_eq!(page_count, 4);
        assert_eq!(unit_test_count, 4);
        assert_eq!(integration_test_count, 4);
        assert_eq!(e2e_test_count, 5);
    }
}
