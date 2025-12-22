use crate::utils::parse_node_types;
use crate::types::{
    Result, CoverageParams, Coverage, CoverageStat, MockStat,
    QueryNodesParams, QueryNodesResponse, NodesResponseItem,
    NodeConcise, Node, HasParams, HasResponse,
    WebError,
};
use shared::Error;
use ast::lang::{graphs::{graph_ops::GraphOps, TestFilters}, NodeType};
use axum::{Json, extract::Query};

#[axum::debug_handler]
pub async fn coverage_handler(Query(params): Query<CoverageParams>) -> Result<Json<Coverage>> {
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

    let totals = graph_ops
        .get_coverage(params.repo.as_deref(), Some(test_filters), params.is_muted)
        .await?;

    Ok(Json(Coverage {
        language: totals.language,
        unit_tests: totals.unit_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        integration_tests: totals.integration_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        e2e_tests: totals.e2e_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        mocks: totals.mocks.map(|s| MockStat {
            total: s.total,
            mocked: s.mocked,
            percent: s.percent,
        }),
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
        .map(|(node_type,  node_data, usage_count, covered, test_count, ref_id, body_len, line_cnt, is_muted)| {
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
        })
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


