#[cfg(all(feature = "neo4j", feature = "fulltest"))]
mod test_endpoints {
    use ast::lang::graphs::graph_ops::GraphOps;
    use ast::lang::graphs::Neo4jGraph;
    use ast::lang::{Graph, Lang, NodeType};
    use ast::repo::{Repo, Repos};
    use test_log::test;
    use tracing::info;

    async fn setup_nextjs_graph() -> GraphOps {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await.unwrap();
        
        if let Err(e) = graph_ops.graph.clear().await {
            eprintln!("Failed to clear graph before setup: {:?}", e);
        }

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
        let graph = repos.build_graphs_inner::<Neo4jGraph>().await.unwrap();
        graph.analysis();

        let (nodes, edges) = graph.get_graph_size();
        info!("Graph built: {} nodes, {} edges", nodes, edges);

        graph_ops
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_endpoint_basic_structure() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        assert!(coverage.unit_tests.is_some());
        assert!(coverage.integration_tests.is_some());
        assert!(coverage.e2e_tests.is_some());

        if let Some(unit) = &coverage.unit_tests {
            assert!(unit.total > 0);
            assert!(unit.total_tests > 0);
            assert!(unit.percent >= 0.0 && unit.percent <= 100.0);
        }

        if let Some(integration) = &coverage.integration_tests {
            assert!(integration.total > 0);
            assert!(integration.total_tests > 0);
            assert!(integration.percent >= 0.0 && integration.percent <= 100.0);
        }

        if let Some(e2e) = &coverage.e2e_tests {
            assert!(e2e.total > 0);
            assert!(e2e.total_tests > 0);
            assert!(e2e.percent >= 0.0 && e2e.percent <= 100.0);
        }
    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_endpoint_test_counts() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        if let Some(unit) = &coverage.unit_tests {
            assert_eq!(unit.total_tests, 4);
            info!(
                "Unit: {} tests, {}/{} covered ({}%)",
                unit.total_tests, unit.covered, unit.total, unit.percent
            );
        }

        if let Some(integration) = &coverage.integration_tests {
            assert_eq!(integration.total_tests, 4);
            info!(
                "Integration: {} tests, {}/{} covered ({}%)",
                integration.total_tests,
                integration.covered,
                integration.total,
                integration.percent
            );
        }

        if let Some(e2e) = &coverage.e2e_tests {
            assert_eq!(e2e.total_tests, 5);
            info!(
                "E2E: {} tests, {}/{} covered ({}%)",
                e2e.total_tests, e2e.covered, e2e.total, e2e.percent
            );
        }

    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_coverage_percentage_calculation() {
        let mut graph_ops = setup_nextjs_graph().await;
        let coverage = graph_ops.get_coverage(None).await.unwrap();

        if let Some(unit) = &coverage.unit_tests {
            let expected_percent = if unit.total > 0 {
                (unit.covered as f64 / unit.total as f64) * 100.0
            } else {
                0.0
            };
            let diff = (unit.percent - expected_percent).abs();
            assert!(diff < 0.01);
        }

        if let Some(integration) = &coverage.integration_tests {
            let expected_percent = if integration.total > 0 {
                (integration.covered as f64 / integration.total as f64) * 100.0
            } else {
                0.0
            };
            let diff = (integration.percent - expected_percent).abs();
            assert!(diff < 0.01);
        }

        if let Some(e2e) = &coverage.e2e_tests {
            let expected_percent = if e2e.total > 0 {
                (e2e.covered as f64 / e2e.total as f64) * 100.0
            } else {
                0.0
            };
            let diff = (e2e.percent - expected_percent).abs();
            assert!(diff < 0.01);
        }

    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_functions_all() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (total_count, results) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        assert!(total_count > 0);
        assert_eq!(results.len(), total_count);

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
            assert!(*covered);
            assert!(*test_count > 0);
        }

        assert!(total_tested > 0, "Should have at least some tested functions");

    }

    #[test(tokio::test(flavor = "multi_thread"))]
    async fn test_nodes_query_functions_untested() {
        let mut graph_ops = setup_nextjs_graph().await;

        let (_total_untested, untested_results) = graph_ops
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
            assert!(!*covered, "All functions in 'untested' filter should not be covered");
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

        let (total_count, _) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, 100, false, None, false, false)
            .await
            .unwrap();

        info!("Total functions in graph: {}", total_count);

        let limit = std::cmp::min(5, total_count);
        let (count1, page1) = graph_ops
            .query_nodes_with_count(NodeType::Function, 0, limit, false, None, false, false)
            .await
            .unwrap();

        let (count2, page2) = graph_ops
            .query_nodes_with_count(NodeType::Function, limit, limit, false, None, false, false)
            .await
            .unwrap();

        assert_eq!(count1, count2);
        assert_eq!(count1, total_count);
        
        if total_count >= limit {
            assert_eq!(page1.len(), limit);
        } else {
            assert_eq!(page1.len(), total_count);
        }

        if total_count > limit {
            assert!(page2.len() > 0);
        }

        let page1_names: Vec<_> = page1.iter().map(|(n, ..)| &n.name).collect();
        let page2_names: Vec<_> = page2.iter().map(|(n, ..)| &n.name).collect();

        for name in &page2_names {
            assert!(!page1_names.contains(name));
        }
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

        let tested_endpoints = endpoint_results
            .iter()
            .filter(|(_, _, covered, _, _, _, _)| *covered)
            .count();

        assert!(tested_endpoints > 0);

        let get_items_tested = endpoint_results
            .iter()
            .any(|(node, _, covered, test_count, _, _, _)| {
                node.name.contains("/api/items")
                    && node.meta.get("verb") == Some(&"GET".to_string())
                    && *covered
                    && *test_count > 0
            });

        assert!(get_items_tested);

    }
}
