use crate::types::{
    Coverage, CoverageParams, CoverageResponse, CoverageStat, HasParams, HasResponse,
    LanguageCoverage, MockStat, Node, NodeConcise, NodesResponseItem, QueryNodesParams,
    QueryNodesResponse, Result, WebError,
};
use crate::utils::parse_node_types;
use ast::lang::{
    graphs::{graph_ops::GraphOps, TestFilters},
    NodeType,
};
use axum::{extract::Query, Json};
use shared::Error;

fn split_at_comma(s: &str) -> Vec<String> {
    s.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Aggregate multiple CoverageStat values into a single combined stat.
fn aggregate_coverage_stats(stats: &[Option<CoverageStat>]) -> Option<CoverageStat> {
    let valid: Vec<_> = stats.iter().filter_map(|s| s.as_ref()).collect();
    if valid.is_empty() {
        return None;
    }

    let total: usize = valid.iter().map(|s| s.total).sum();
    let total_tests: usize = valid.iter().map(|s| s.total_tests).sum();
    let covered: usize = valid.iter().map(|s| s.covered).sum();
    let total_lines: usize = valid.iter().map(|s| s.total_lines).sum();
    let covered_lines: usize = valid.iter().map(|s| s.covered_lines).sum();

    Some(CoverageStat {
        total,
        total_tests,
        covered,
        percent: if total > 0 {
            (covered as f64 / total as f64) * 100.0
        } else {
            0.0
        },
        total_lines,
        covered_lines,
        line_percent: if total_lines > 0 {
            (covered_lines as f64 / total_lines as f64) * 100.0
        } else {
            0.0
        },
    })
}

/// Aggregate multiple MockStat values into a single combined stat.
fn aggregate_mock_stats(stats: &[Option<MockStat>]) -> Option<MockStat> {
    let valid: Vec<_> = stats.iter().filter_map(|s| s.as_ref()).collect();
    if valid.is_empty() {
        return None;
    }

    let total: usize = valid.iter().map(|s| s.total).sum();
    let mocked: usize = valid.iter().map(|s| s.mocked).sum();

    Some(MockStat {
        total,
        mocked,
        percent: if total > 0 {
            (mocked as f64 / total as f64) * 100.0
        } else {
            0.0
        },
    })
}

#[axum::debug_handler]
pub async fn coverage_handler(
    Query(params): Query<CoverageParams>,
) -> Result<Json<CoverageResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let ignore_dirs = params
        .ignore_dirs
        .as_ref()
        .map(|s| split_at_comma(s))
        .unwrap_or_default();

    let languages = params.language.as_ref().map(|s| split_at_comma(s));

    let test_filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: params.regex.clone(),
        ignore_dirs,
        languages,
    };

    let graph_coverages = graph_ops
        .get_coverage(params.repo.as_deref(), Some(test_filters), params.is_muted)
        .await?;

    // Convert to per-language Coverage structs
    let coverages: Vec<Coverage> = graph_coverages.into_iter().map(Coverage::from).collect();

    // Build per-language breakdown
    let languages: Vec<LanguageCoverage> = coverages
        .iter()
        .map(|c| LanguageCoverage {
            name: c.language.clone().unwrap_or_default(),
            unit_tests: c.unit_tests.clone(),
            integration_tests: c.integration_tests.clone(),
            e2e_tests: c.e2e_tests.clone(),
            mocks: c.mocks.clone(),
        })
        .collect();

    // Aggregate totals across all languages
    let unit_tests = aggregate_coverage_stats(
        &coverages
            .iter()
            .map(|c| c.unit_tests.clone())
            .collect::<Vec<_>>(),
    );
    let integration_tests = aggregate_coverage_stats(
        &coverages
            .iter()
            .map(|c| c.integration_tests.clone())
            .collect::<Vec<_>>(),
    );
    let e2e_tests = aggregate_coverage_stats(
        &coverages
            .iter()
            .map(|c| c.e2e_tests.clone())
            .collect::<Vec<_>>(),
    );
    let mocks = aggregate_mock_stats(
        &coverages
            .iter()
            .map(|c| c.mocks.clone())
            .collect::<Vec<_>>(),
    );

    Ok(Json(CoverageResponse {
        unit_tests,
        integration_tests,
        e2e_tests,
        mocks,
        languages,
    }))
}

#[axum::debug_handler]
pub async fn nodes_handler(
    Query(params): Query<QueryNodesParams>,
) -> Result<Json<QueryNodesResponse>> {
    let node_types = parse_node_types(&params.node_type).map_err(|e| WebError(e))?;
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(10).min(100);
    let sort_by_test_count = params.sort.as_deref().unwrap_or("test_count") == "test_count";
    let coverage_filter = params.coverage.as_deref();
    let concise = params.concise.unwrap_or(true);
    let body_length = params.body_length.unwrap_or(false);
    let line_count = params.line_count.unwrap_or(false);
    let is_muted = params.is_muted;

    if let Some(coverage) = coverage_filter {
        if !matches!(coverage, "tested" | "untested" | "all") {
            return Err(WebError(shared::Error::Custom(
                "Invalid coverage parameter. Must be 'tested', 'untested', or 'all'".into(),
            )));
        }
    }

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let test_filters = TestFilters {
        unit_regexes: params
            .unit_regexes
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
        integration_regexes: params
            .integration_regexes
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
        e2e_regexes: params
            .e2e_regexes
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
        target_regex: params.regex.clone(),
        ignore_dirs: params
            .ignore_dirs
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
        languages: None,
    };

    let (total_count, results) = graph_ops
        .query_nodes_with_count(
            &node_types,
            offset,
            limit,
            sort_by_test_count,
            coverage_filter,
            body_length,
            line_count,
            params.repo.as_deref(),
            Some(test_filters),
            params.search.as_deref(),
            is_muted,
        )
        .await?;

    let items: Vec<NodesResponseItem> = results
        .into_iter()
        .map(
            |(
                node_type,
                node_data,
                usage_count,
                covered,
                test_count,
                ref_id,
                body_len,
                line_cnt,
                is_muted,
            )| {
                let verb = if node_type == NodeType::Endpoint {
                    node_data.meta.get("verb").cloned()
                } else {
                    None
                };

                if concise {
                    NodesResponseItem::Concise(NodeConcise {
                        node_type: node_type.to_string(),
                        name: node_data.name.clone(),
                        file: node_data.file.clone(),
                        ref_id,
                        weight: usage_count,
                        test_count,
                        covered,
                        body_length: body_len,
                        line_count: line_cnt,
                        verb,
                        start: node_data.start,
                        end: node_data.end,
                        meta: node_data.meta,
                        is_muted,
                    })
                } else {
                    NodesResponseItem::Full(Node {
                        node_type: node_type.to_string(),
                        ref_id,
                        weight: usage_count,
                        test_count,
                        covered,
                        properties: node_data,
                        body_length: body_len,
                        line_count: line_cnt,
                        is_muted,
                    })
                }
            },
        )
        .collect();

    let total_returned = items.len();
    let total_pages = if limit > 0 {
        (total_count + limit - 1) / limit
    } else {
        0
    };
    let current_page = if limit > 0 { (offset / limit) + 1 } else { 0 };

    Ok(Json(QueryNodesResponse {
        items,
        total_returned,
        total_count,
        total_pages,
        current_page,
    }))
}

#[axum::debug_handler]
pub async fn has_handler(Query(params): Query<HasParams>) -> Result<Json<HasResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let node_type = match params.node_type.to_lowercase().as_str() {
        "function" => NodeType::Function,
        "endpoint" => NodeType::Endpoint,
        _ => return Err(WebError(Error::Custom("invalid node_type".into()))),
    };
    println!(
        "[/tests/has] node_type={:?} name={:?} file={:?} start={:?} root={:?} tests={:?}",
        node_type, params.name, params.file, params.start, params.root, params.tests
    );
    let covered = graph_ops
        .has_coverage(
            node_type,
            &params.name,
            &params.file,
            params.start,
            params.root.as_deref(),
            params.tests.as_deref(),
        )
        .await?;
    Ok(Json(HasResponse { covered }))
}
