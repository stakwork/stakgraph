use crate::types::{
    CoverageParams, MockStat, QueryNodesParams, Result, TransitiveCoverage, TransitiveCoverageStat,
    TransitiveNode, TransitiveNodeConcise, TransitiveNodesResponse, TransitiveNodesResponseItem,
    WebError,
};
use crate::utils::parse_node_types;
use ast::lang::{
    graphs::{graph_ops::GraphOps, TestFilters},
    NodeType,
};
use axum::{extract::Query, Json};

#[axum::debug_handler]
pub async fn transitive_stats_handler(
    Query(params): Query<CoverageParams>,
) -> Result<Json<TransitiveCoverage>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let test_filters = TestFilters {
        unit_regexes: vec![],
        integration_regexes: vec![],
        e2e_regexes: vec![],
        target_regex: params.regex.clone(),
        ignore_dirs: params
            .ignore_dirs
            .as_ref()
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default(),
    };

    let graph_coverage = graph_ops
        .get_transitive_coverage(params.repo.as_deref(), Some(test_filters), params.is_muted)
        .await?;

    let coverage = TransitiveCoverage {
        language: graph_coverage.language,
        unit_tests: graph_coverage.unit_tests.map(|s| TransitiveCoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            direct_covered: s.direct_covered,
            transitive_covered: s.transitive_covered,
            direct_percent: s.direct_percent,
            transitive_percent: s.transitive_percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        integration_tests: graph_coverage
            .integration_tests
            .map(|s| TransitiveCoverageStat {
                total: s.total,
                total_tests: s.total_tests,
                direct_covered: s.direct_covered,
                transitive_covered: s.transitive_covered,
                direct_percent: s.direct_percent,
                transitive_percent: s.transitive_percent,
                total_lines: s.total_lines,
                covered_lines: s.covered_lines,
                line_percent: s.line_percent,
            }),
        e2e_tests: graph_coverage.e2e_tests.map(|s| TransitiveCoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            direct_covered: s.direct_covered,
            transitive_covered: s.transitive_covered,
            direct_percent: s.direct_percent,
            transitive_percent: s.transitive_percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        mocks: graph_coverage.mocks.map(|m| MockStat {
            total: m.total,
            mocked: m.mocked,
            percent: m.percent,
        }),
    };

    Ok(Json(coverage))
}

#[axum::debug_handler]
pub async fn transitive_nodes_handler(
    Query(params): Query<QueryNodesParams>,
) -> Result<Json<TransitiveNodesResponse>> {
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
    };

    let (total_count, results) = graph_ops
        .query_transitive_nodes_with_count(
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

    let items: Vec<TransitiveNodesResponseItem> = results
        .into_iter()
        .map(
            |(
                node_type,
                node_data,
                usage_count,
                direct_covered,
                transitive_covered,
                direct_test_count,
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
                    TransitiveNodesResponseItem::Concise(TransitiveNodeConcise {
                        node_type: node_type.to_string(),
                        name: node_data.name.clone(),
                        file: node_data.file.clone(),
                        ref_id,
                        weight: usage_count,
                        direct_test_count,
                        direct_covered,
                        transitive_covered,
                        body_length: body_len,
                        line_count: line_cnt,
                        verb,
                        start: node_data.start,
                        end: node_data.end,
                        meta: node_data.meta,
                        is_muted,
                    })
                } else {
                    TransitiveNodesResponseItem::Full(TransitiveNode {
                        node_type: node_type.to_string(),
                        ref_id,
                        weight: usage_count,
                        direct_test_count,
                        direct_covered,
                        transitive_covered,
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

    Ok(Json(TransitiveNodesResponse {
        items,
        total_returned,
        total_count,
        total_pages,
        current_page,
    }))
}
