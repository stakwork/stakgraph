use crate::lang::graphs::neo4j::helpers::*;
use crate::lang::{NodeType, TestFilters};
use neo4rs::{BoltMap, BoltType};

const MAX_TRAVERSAL_DEPTH: u32 = 5;

/// Query transitive coverage stats for a specific test type
/// - test_type: "unit" -> UnitTest -> Function (PLUS functions from integration-tested endpoints!)
/// - test_type: "integration" -> IntegrationTest -> Endpoint  
/// - test_type: "e2e" -> E2eTest -> Page
pub fn query_transitive_coverage_stats_by_type(
    test_type: &str,
    repo: Option<&str>,
    test_filters: Option<TestFilters>,
    is_muted: Option<bool>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let (test_label, target_type) = match test_type {
        "integration" => ("IntegrationTest", NodeType::Endpoint),
        "e2e" => ("E2etest", NodeType::Page),
        _ => ("UnitTest", NodeType::Function), // default to unit
    };

    let node_types = vec![target_type.clone()];
    let node_types_list = node_types
        .iter()
        .map(|n| BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

    let has_function_type = target_type == NodeType::Function;

    let function_specific_filters = if has_function_type {
        format!("AND ({})", unique_functions_filters().join(" AND "))
    } else {
        "".to_string() // No special filters for Endpoint/Page
    };

    let repo_filter = if let Some(r) = repo {
        if r.is_empty() || r == "all" {
            String::new()
        } else if r.contains(',') {
            let repos: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
            let conditions: Vec<String> = repos
                .iter()
                .map(|repo_path| format!("n.file STARTS WITH '{}'", repo_path))
                .collect();
            format!("AND ({})", conditions.join(" OR "))
        } else {
            format!("AND n.file STARTS WITH '{}'", r)
        }
    } else {
        String::new()
    };

    let ignore_dirs_filter = if !ignore_dirs.is_empty() {
        let conditions: Vec<String> = ignore_dirs
            .iter()
            .map(|dir| {
                format!(
                    "(NOT n.file CONTAINS '/{dir}/' AND NOT n.file =~ '.*/{dir}$')",
                    dir = dir
                )
            })
            .collect();
        format!("AND {}", conditions.join(" AND "))
    } else {
        String::new()
    };

    let regex_filter = test_filters
        .as_ref()
        .and_then(|f| f.target_regex.as_deref())
        .map(|pattern| format!("AND n.file =~ '{}'", pattern))
        .unwrap_or_default();

    let is_muted_filter = match is_muted {
        Some(true) => "AND (n.is_muted = true OR n.is_muted = 'true')",
        Some(false) => "AND (n.is_muted IS NULL OR (n.is_muted <> true AND n.is_muted <> 'true'))",
        None => "",
    };

    // For unit tests, include functions called by ANY test type PLUS HANDLER functions from endpoints
    let query = if test_type == "unit" {
        format!(
            "
             MATCH (n:Function)
             WHERE n.node_key IS NOT NULL
             {} {} {} {} {}
             WITH COLLECT(DISTINCT n) AS allNodes

             // Find ALL functions directly called by ANY test type
             OPTIONAL MATCH (test)-[:CALLS]->(directFromTest:Function)
             WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
             AND directFromTest IN allNodes
             WITH allNodes, COLLECT(DISTINCT directFromTest) AS directlyCalledByTests

             // Find functions that are HANDLERs of endpoints called by integration tests
             OPTIONAL MATCH (intTest:IntegrationTest)-[:CALLS]->(endpoint:Endpoint)-[:HANDLER]->(handlerFunc:Function)
             WHERE handlerFunc IN allNodes
             WITH allNodes, directlyCalledByTests, COLLECT(DISTINCT handlerFunc) AS handlerFunctions

             // Combine: directly tested = called by tests + handler functions
             WITH allNodes, 
                  directlyCalledByTests + [h IN handlerFunctions WHERE NOT h IN directlyCalledByTests] AS directlyTested

             // Count unit tests for the total_tests stat
             OPTIONAL MATCH (unitTest:UnitTest)-[:CALLS]->(directForCount:Function)
             WHERE directForCount IN allNodes
             WITH allNodes, directlyTested, count(DISTINCT unitTest) AS total_tests

             // Traverse from all directly tested functions to find transitively covered
             UNWIND CASE WHEN size(directlyTested) = 0 THEN [null] ELSE directlyTested END AS d
             OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|OPERAND|RENDERS|HANDLER*1..{max_depth}]->(transitive:Function)
             WHERE transitive IN allNodes
             WITH allNodes, directlyTested, total_tests, COLLECT(DISTINCT transitive) AS transitiveRaw

             // Combine: transitive = direct + reachable
             WITH allNodes, directlyTested, total_tests,
                  directlyTested + [t IN transitiveRaw WHERE t IS NOT NULL AND NOT t IN directlyTested] AS transitivelyCovered

             // Compute stats
             WITH size(allNodes) AS total,
                  total_tests,
                  size(directlyTested) AS direct_covered,
                  size(transitivelyCovered) AS transitive_covered,
                  REDUCE(s = 0, n IN allNodes | s + COALESCE(n.end - n.start + 1, 0)) AS total_lines,
                  REDUCE(s = 0, n IN transitivelyCovered | s + COALESCE(n.end - n.start + 1, 0)) AS covered_lines

             RETURN total, total_tests, direct_covered, transitive_covered, total_lines, covered_lines",
            function_specific_filters,
            repo_filter, 
            ignore_dirs_filter, 
            regex_filter, 
            is_muted_filter,
            max_depth = MAX_TRAVERSAL_DEPTH
        )
    } else {
        // For integration/e2e, use the original logic
        format!(
            "
             MATCH (n:{target_type})
             WHERE n.node_key IS NOT NULL
             {} {} {} {} {}

             WITH COLLECT(DISTINCT n) AS allNodes

             OPTIONAL MATCH (test:{test_label})-[:CALLS]->(direct)
             WHERE direct IN allNodes
             WITH allNodes, COLLECT(DISTINCT direct) AS directlyTested

             OPTIONAL MATCH (test:{test_label})-[:CALLS]->(directForCount)
             WHERE directForCount IN allNodes
             WITH allNodes, directlyTested, count(DISTINCT test) AS total_tests

             UNWIND CASE WHEN size(directlyTested) = 0 THEN [null] ELSE directlyTested END AS d
             OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{max_depth}]->(transitive)
             WHERE transitive IN allNodes
             WITH allNodes, directlyTested, total_tests, COLLECT(DISTINCT transitive) AS transitiveRaw

             WITH allNodes, directlyTested, total_tests,
                  directlyTested + [t IN transitiveRaw WHERE t IS NOT NULL AND NOT t IN directlyTested] AS transitivelyCovered

             WITH size(allNodes) AS total,
                  total_tests,
                  size(directlyTested) AS direct_covered,
                  size(transitivelyCovered) AS transitive_covered,
                  REDUCE(s = 0, n IN allNodes | s + COALESCE(n.end - n.start + 1, 0)) AS total_lines,
                  REDUCE(s = 0, n IN transitivelyCovered | s + COALESCE(n.end - n.start + 1, 0)) AS covered_lines

             RETURN total, total_tests, direct_covered, transitive_covered, total_lines, covered_lines",
            function_specific_filters,
            repo_filter, 
            ignore_dirs_filter, 
            regex_filter, 
            is_muted_filter,
            target_type = target_type,
            test_label = test_label,
            max_depth = MAX_TRAVERSAL_DEPTH
        )
    };

    (query, params)
}

/// Original query for backward compatibility (uses Function type with all test types)
pub fn query_transitive_coverage_stats(
    node_types: &[NodeType],
    repo: Option<&str>,
    test_filters: Option<TestFilters>,
    is_muted: Option<bool>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let node_types_list = node_types
        .iter()
        .map(|n| BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

    let has_function_type = node_types.iter().any(|nt| nt == &NodeType::Function);

    let function_specific_filters = if has_function_type {
        format!("AND ({})", unique_functions_filters().join(" AND "))
    } else {
        "AND (n.body IS NOT NULL AND n.body <> '')".to_string()
    };

    let repo_filter = if let Some(r) = repo {
        if r.is_empty() || r == "all" {
            String::new()
        } else if r.contains(',') {
            let repos: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
            let conditions: Vec<String> = repos
                .iter()
                .map(|repo_path| format!("n.file STARTS WITH '{}'", repo_path))
                .collect();
            format!("AND ({})", conditions.join(" OR "))
        } else {
            format!("AND n.file STARTS WITH '{}'", r)
        }
    } else {
        String::new()
    };

    let ignore_dirs_filter = if !ignore_dirs.is_empty() {
        let conditions: Vec<String> = ignore_dirs
            .iter()
            .map(|dir| {
                format!(
                    "(NOT n.file CONTAINS '/{dir}/' AND NOT n.file =~ '.*/{dir}$')",
                    dir = dir
                )
            })
            .collect();
        format!("AND {}", conditions.join(" AND "))
    } else {
        String::new()
    };

    let regex_filter = test_filters
        .as_ref()
        .and_then(|f| f.target_regex.as_deref())
        .map(|pattern| format!("AND n.file =~ '{}'", pattern))
        .unwrap_or_default();

    let is_muted_filter = match is_muted {
        Some(true) => "AND (n.is_muted = true OR n.is_muted = 'true')",
        Some(false) => "AND (n.is_muted IS NULL OR (n.is_muted <> true AND n.is_muted <> 'true'))",
        None => "",
    };

    let query = format!(
        "
         MATCH (n)
         WHERE ANY(label IN labels(n) WHERE label IN $node_types)
         {} {} {} {} {}
         WITH COLLECT(DISTINCT n) AS allNodes

         OPTIONAL MATCH (test)-[:CALLS]->(direct)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND direct IN allNodes
         WITH allNodes, COLLECT(DISTINCT direct) AS directlyTested
         UNWIND CASE WHEN size(directlyTested) = 0 THEN [null] ELSE directlyTested END AS d
         OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{}]->(transitive)
         WHERE transitive IN allNodes
         WITH allNodes, directlyTested, COLLECT(DISTINCT transitive) AS transitiveRaw

         WITH allNodes, directlyTested, 
              directlyTested + [t IN transitiveRaw WHERE t IS NOT NULL AND NOT t IN directlyTested] AS transitivelyCovered

         WITH size(allNodes) AS total,
              size(directlyTested) AS direct_covered,
              size(transitivelyCovered) AS transitive_covered,
              REDUCE(s = 0, n IN allNodes | s + COALESCE(n.end - n.start + 1, 0)) AS total_lines,
              REDUCE(s = 0, n IN transitivelyCovered | s + COALESCE(n.end - n.start + 1, 0)) AS covered_lines

         RETURN total, direct_covered, transitive_covered, total_lines, covered_lines",
        function_specific_filters,
        repo_filter, 
        ignore_dirs_filter, 
        regex_filter, 
        is_muted_filter,
        MAX_TRAVERSAL_DEPTH
    );

    (query, params)
}

pub fn query_transitive_nodes_with_count(
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
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_int(&mut params, "offset", offset as i64);
    boltmap_insert_int(&mut params, "limit", limit as i64);

    let node_types_list = node_types
        .iter()
        .map(|n| BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    if let Some(search_term) = search {
        boltmap_insert_str(&mut params, "search", search_term);
    }

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

    let has_function_type = node_types.iter().any(|nt| nt == &NodeType::Function);

    let function_specific_filters = if has_function_type {
        format!("AND ({})", unique_functions_filters().join(" AND "))
    } else {
        "AND (n.body IS NOT NULL AND n.body <> '')".to_string()
    };

    let repo_filter = if let Some(r) = repo {
        if r.is_empty() || r == "all" {
            String::new()
        } else if r.contains(',') {
            let repos: Vec<&str> = r.split(',').map(|s| s.trim()).collect();
            let conditions: Vec<String> = repos
                .iter()
                .map(|repo_path| format!("n.file STARTS WITH '{}'", repo_path))
                .collect();
            format!("AND ({})", conditions.join(" OR "))
        } else {
            format!("AND n.file STARTS WITH '{}'", r)
        }
    } else {
        String::new()
    };

    let ignore_dirs_filter = if !ignore_dirs.is_empty() {
        let conditions: Vec<String> = ignore_dirs
            .iter()
            .map(|dir| {
                format!(
                    "(NOT n.file CONTAINS '/{dir}/' AND NOT n.file =~ '.*/{dir}$')",
                    dir = dir
                )
            })
            .collect();
        format!("AND {}", conditions.join(" AND "))
    } else {
        String::new()
    };

    let regex_filter = test_filters
        .as_ref()
        .and_then(|f| f.target_regex.as_deref())
        .map(|pattern| format!("AND n.file =~ '{}'", pattern))
        .unwrap_or_default();

    let search_filter = search
        .map(|_| "AND toLower(n.name) CONTAINS toLower($search)".to_string())
        .unwrap_or_default();

    let is_muted_filter = match is_muted {
        Some(true) => "AND (n.is_muted = true OR n.is_muted = 'true')",
        Some(false) => "AND (n.is_muted IS NULL OR (n.is_muted <> true AND n.is_muted <> 'true'))",
        None => "",
    };

    let order_clause = if body_length {
        "ORDER BY item.body_length DESC, item.node.name ASC, item.node.file ASC"
    } else if line_count {
        "ORDER BY item.line_count DESC, item.node.name ASC, item.node.file ASC"
    } else if sort_by_test_count {
        "ORDER BY item.direct_test_count DESC, item.node.name ASC, item.node.file ASC"
    } else {
        "ORDER BY item.node.name ASC, item.node.file ASC"
    };

    let coverage_where = match coverage_filter {
        Some("tested") => "WHERE item.is_transitive_covered = true",
        Some("untested") => "WHERE item.is_transitive_covered = false",
        _ => "",
    };

    let query = format!(
        "
         MATCH (n)
         WHERE ANY(label IN labels(n) WHERE label IN $node_types)
         {} {} {} {} {} {}
         WITH COLLECT(DISTINCT n) AS allNodes

         OPTIONAL MATCH (test)-[:CALLS]->(direct)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND direct IN allNodes
         WITH allNodes, 
              COLLECT(DISTINCT direct) AS directlyTested,
              COLLECT({{node: direct, test: test}}) AS testPairs

         UNWIND CASE WHEN size(directlyTested) = 0 THEN [null] ELSE directlyTested END AS d
         OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{}]->(transitive)
         WHERE transitive IN allNodes
         WITH allNodes, directlyTested, testPairs,
              COLLECT(DISTINCT transitive) AS transitiveRaw

         WITH allNodes, directlyTested, testPairs,
              directlyTested + [t IN transitiveRaw WHERE t IS NOT NULL AND NOT t IN directlyTested] AS transitivelyCovered

         UNWIND allNodes AS n
         WITH n, directlyTested, transitivelyCovered, testPairs,
              n IN transitivelyCovered AS is_transitive_covered,
              n IN directlyTested AS is_direct_covered,
              size([pair IN testPairs WHERE pair.node = n]) AS direct_test_count

         OPTIONAL MATCH (caller)-[:CALLS]->(n)
         WITH n, is_transitive_covered, is_direct_covered, direct_test_count,
              count(DISTINCT caller) AS usage_count
         WITH collect({{
             node: n,
             usage_count: usage_count,
             is_direct_covered: is_direct_covered,
             is_transitive_covered: is_transitive_covered,
             direct_test_count: direct_test_count,
             body_length: size(n.body),
             line_count: (n.end - n.start + 1),
             is_muted: CASE 
                 WHEN n.is_muted = true OR n.is_muted = 'true' THEN true
                 WHEN n.is_muted = false OR n.is_muted = 'false' THEN false
                 ELSE null 
             END
         }}) AS all_items

         // Step 8: Filter, sort, and paginate
         WITH [item IN all_items {}] AS filtered_items
         WITH filtered_items, size(filtered_items) AS total_count
         UNWIND filtered_items AS item
         WITH item, total_count
         {}
         WITH collect(item) AS sorted_items, total_count
         RETURN 
             total_count,
             sorted_items[$offset..($offset + $limit)] AS items",
        function_specific_filters,
        repo_filter,
        ignore_dirs_filter,
        regex_filter,
        search_filter,
        is_muted_filter,
        MAX_TRAVERSAL_DEPTH,
        coverage_where,
        order_clause
    );

    (query, params)
}
