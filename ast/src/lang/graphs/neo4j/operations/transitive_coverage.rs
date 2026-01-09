use shared::Result;

use crate::lang::graphs::neo4j::executor::bind_parameters;
use crate::lang::graphs::neo4j::queries::transitive_coverage::*;
use crate::lang::graphs::operations::coverage::MockStat as CoverageMockStat;
use crate::lang::{graph_ops::GraphOps, NodeData, NodeType, TestFilters};

#[derive(Debug, Clone)]
pub struct TransitiveGraphCoverage {
    pub language: Option<String>,
    pub unit_tests: Option<TransitiveCoverageStat>,
    pub integration_tests: Option<TransitiveCoverageStat>,
    pub e2e_tests: Option<TransitiveCoverageStat>,
    pub mocks: Option<CoverageMockStat>,
}

#[derive(Debug, Clone)]
pub struct TransitiveCoverageStat {
    pub total: usize,
    pub total_tests: usize,
    pub direct_covered: usize,
    pub transitive_covered: usize,
    pub direct_percent: f64,
    pub transitive_percent: f64,
    pub total_lines: usize,
    pub covered_lines: usize,
    pub line_percent: f64,
}

impl GraphOps {
    pub async fn get_transitive_coverage(
        &mut self,
        repo: Option<&str>,
        test_filters: Option<TestFilters>,
        is_muted: Option<bool>,
    ) -> Result<TransitiveGraphCoverage> {
        self.graph.ensure_connected().await?;

        let node_types = vec![NodeType::Function];
        let (query_str, params) =
            query_transitive_coverage_stats(&node_types, repo, test_filters.clone(), is_muted);

        let conn = self.graph.ensure_connected().await?;
        let query_obj = bind_parameters(&query_str, params);
        let mut result = conn.execute(query_obj).await?;

        let mut total = 0usize;
        let mut direct_covered = 0usize;
        let mut transitive_covered = 0usize;
        let mut total_lines = 0usize;
        let mut covered_lines = 0usize;

        if let Some(row) = result.next().await? {
            total = row.get::<i64>("total").unwrap_or(0) as usize;
            direct_covered = row.get::<i64>("direct_covered").unwrap_or(0) as usize;
            transitive_covered = row.get::<i64>("transitive_covered").unwrap_or(0) as usize;
            total_lines = row.get::<i64>("total_lines").unwrap_or(0) as usize;
            covered_lines = row.get::<i64>("covered_lines").unwrap_or(0) as usize;
        }

        let direct_percent = if total > 0 {
            (direct_covered as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let transitive_percent = if total > 0 {
            (transitive_covered as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let line_percent = if total_lines > 0 {
            (covered_lines as f64 / total_lines as f64) * 100.0
        } else {
            0.0
        };

        let unit_tests_count = self
            .graph
            .find_nodes_by_type_async(NodeType::UnitTest)
            .await
            .len();

        let unit_stat = if total > 0 {
            Some(TransitiveCoverageStat {
                total,
                total_tests: unit_tests_count,
                direct_covered,
                transitive_covered,
                direct_percent: (direct_percent * 100.0).round() / 100.0,
                transitive_percent: (transitive_percent * 100.0).round() / 100.0,
                total_lines,
                covered_lines,
                line_percent: (line_percent * 100.0).round() / 100.0,
            })
        } else {
            None
        };

        Ok(TransitiveGraphCoverage {
            language: Some("typescript".to_string()),
            unit_tests: unit_stat,
            integration_tests: None,
            e2e_tests: None,
            mocks: None,
        })
    }

    pub async fn query_transitive_nodes_with_count(
        &mut self,
        node_types: &[NodeType],
        offset: usize,
        limit: usize,
        sort_by_test_count: bool,
        coverage_filter: Option<&str>,
        body_length: bool,
        line_count: bool,
        repo: Option<&str>,
        test_filters: Option<TestFilters>,
        search: Option<&str>,
        is_muted: Option<bool>,
    ) -> Result<(
        usize,
        Vec<(
            NodeType,
            NodeData,
            usize,        // usage_count
            bool,         // direct_covered
            bool,         // transitive_covered
            usize,        // direct_test_count
            String,       // ref_id
            Option<i64>,  // body_len
            Option<i64>,  // line_cnt
            Option<bool>, // is_muted
        )>,
    )> {
        self.graph.ensure_connected().await?;

        let (query_str, params) = query_transitive_nodes_with_count(
            node_types,
            offset,
            limit,
            sort_by_test_count,
            coverage_filter,
            body_length,
            line_count,
            repo,
            test_filters,
            search,
            is_muted,
        );

        let conn = self.graph.ensure_connected().await?;
        let query_obj = bind_parameters(&query_str, params);
        let mut result = conn.execute(query_obj).await?;

        let mut total_count = 0usize;
        let mut items: Vec<(
            NodeType,
            NodeData,
            usize,
            bool,
            bool,
            usize,
            String,
            Option<i64>,
            Option<i64>,
            Option<bool>,
        )> = Vec::new();

        if let Some(row) = result.next().await? {
            total_count = row.get::<i64>("total_count").unwrap_or(0) as usize;

            if let Ok(items_list) = row.get::<Vec<neo4rs::BoltMap>>("items") {
                for item in items_list {
                    let node_bolt = match item.get::<neo4rs::Node>("node") {
                        Ok(n) => n,
                        Err(_) => continue,
                    };

                    let node_data: NodeData = match (&node_bolt).try_into() {
                        Ok(nd) => nd,
                        Err(_) => continue,
                    };

                    let labels: Vec<String> =
                        node_bolt.labels().iter().map(|s| s.to_string()).collect();
                    let node_type = labels
                        .iter()
                        .find_map(|l| l.parse::<NodeType>().ok())
                        .unwrap_or(NodeType::Function);

                    let usage_count = item.get::<i64>("usage_count").unwrap_or(0) as usize;
                    let is_direct = item.get::<bool>("is_direct_covered").unwrap_or(false);
                    let is_transitive = item.get::<bool>("is_transitive_covered").unwrap_or(false);
                    let test_count = item.get::<i64>("direct_test_count").unwrap_or(0) as usize;
                    let body_len = item.get::<i64>("body_length").ok();
                    let line_cnt = item.get::<i64>("line_count").ok();
                    let is_muted_val = item.get::<bool>("is_muted").ok();
                    let ref_id = node_bolt
                        .get::<String>("ref_id")
                        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());

                    items.push((
                        node_type,
                        node_data,
                        usage_count,
                        is_direct,
                        is_transitive,
                        test_count,
                        ref_id,
                        body_len,
                        line_cnt,
                        is_muted_val,
                    ));
                }
            }
        }

        Ok((total_count, items))
    }
}
