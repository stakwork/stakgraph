use crate::lang::graphs::neo4j::helpers::*;
use crate::lang::{NodeType, TestFilters};
use neo4rs::BoltMap;

const MAX_TRAVERSAL_DEPTH: u32 = 5;

pub fn query_directly_tested_nodes(
    node_types: &[NodeType],
    repo: Option<&str>,
    test_filters: Option<TestFilters>,
    is_muted: Option<bool>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let node_types_list = node_types
        .iter()
        .map(|n| neo4rs::BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

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
        "MATCH (test)-[:CALLS]->(n)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND ANY(label IN labels(n) WHERE label IN $node_types)
         {} {} {} {}
         RETURN DISTINCT n.node_key AS node_key, n.name AS name, n.file AS file, n.start AS start, n.end AS end",
        repo_filter, ignore_dirs_filter, regex_filter, is_muted_filter
    );

    (query, params)
}

pub fn query_transitive_nodes(
    node_types: &[NodeType],
    repo: Option<&str>,
    test_filters: Option<TestFilters>,
    is_muted: Option<bool>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let node_types_list = node_types
        .iter()
        .map(|n| neo4rs::BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

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

    let transitive_is_muted_filter = match is_muted {
        Some(true) => "AND (transitive.is_muted = true OR transitive.is_muted = 'true')",
        Some(false) => "AND (transitive.is_muted IS NULL OR (transitive.is_muted <> true AND transitive.is_muted <> 'true'))",
        None => "",
    };

    let query = format!(
        "// Step 1: Find directly tested nodes
         MATCH (test)-[:CALLS]->(direct)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND ANY(label IN labels(direct) WHERE label IN $node_types)
         {} {} {} {}
         WITH COLLECT(DISTINCT direct) AS directlyTested

         // Step 2: Traverse outward from directly tested nodes
         UNWIND directlyTested AS d
         OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{}]->(transitive)
         WHERE ANY(label IN labels(transitive) WHERE label IN $node_types)
         {}
         WITH directlyTested, COLLECT(DISTINCT transitive) AS transitiveNodes

         // Step 3: Combine and dedupe
         WITH directlyTested + [t IN transitiveNodes WHERE NOT t IN directlyTested] AS allCovered
         UNWIND allCovered AS n
         RETURN DISTINCT n.node_key AS node_key, n.name AS name, n.file AS file, n.start AS start, n.end AS end",
        repo_filter.replace("n.", "direct."),
        ignore_dirs_filter.replace("n.", "direct."),
        regex_filter.replace("n.", "direct."),
        is_muted_filter.replace("n.", "direct."),
        MAX_TRAVERSAL_DEPTH,
        transitive_is_muted_filter
    );

    (query, params)
}

pub fn query_transitive_coverage_stats(
    node_types: &[NodeType],
    repo: Option<&str>,
    test_filters: Option<TestFilters>,
    is_muted: Option<bool>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let node_types_list = node_types
        .iter()
        .map(|n| neo4rs::BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

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
        "// Step 1: Get all nodes in scope
         MATCH (n)
         WHERE ANY(label IN labels(n) WHERE label IN $node_types)
         AND n.body IS NOT NULL AND n.body <> ''
         {} {} {} {}
         WITH COLLECT(DISTINCT n) AS allNodes

         // Step 2: Find directly tested nodes
         MATCH (test)-[:CALLS]->(direct)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND direct IN allNodes
         WITH allNodes, COLLECT(DISTINCT direct) AS directlyTested

         // Step 3: Find transitively covered nodes
         UNWIND directlyTested AS d
         OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{}]->(transitive)
         WHERE transitive IN allNodes
         WITH allNodes, directlyTested, COLLECT(DISTINCT transitive) AS transitiveRaw

         // Step 4: Combine transitive (includes direct + transitive)
         WITH allNodes, directlyTested, 
              directlyTested + [t IN transitiveRaw WHERE NOT t IN directlyTested] AS transitivelyCovered

         // Step 5: Compute stats
         WITH size(allNodes) AS total,
              size(directlyTested) AS direct_covered,
              size(transitivelyCovered) AS transitive_covered,
              REDUCE(s = 0, n IN allNodes | s + COALESCE(n.end - n.start + 1, 0)) AS total_lines,
              REDUCE(s = 0, n IN transitivelyCovered | s + COALESCE(n.end - n.start + 1, 0)) AS covered_lines

         RETURN total, direct_covered, transitive_covered, total_lines, covered_lines",
        repo_filter, ignore_dirs_filter, regex_filter, is_muted_filter,
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
        .map(|n| neo4rs::BoltType::String(n.to_string().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "node_types", node_types_list);

    if let Some(search_term) = search {
        boltmap_insert_str(&mut params, "search", search_term);
    }

    let ignore_dirs = test_filters
        .as_ref()
        .map(|f| f.ignore_dirs.clone())
        .unwrap_or_default();

    let order_clause = if body_length {
        "ORDER BY size(n.body) DESC, n.name ASC, n.file ASC"
    } else if line_count {
        "ORDER BY (n.end - n.start) DESC, n.name ASC, n.file ASC"
    } else if sort_by_test_count {
        "ORDER BY direct_test_count DESC, n.name ASC, n.file ASC"
    } else {
        "ORDER BY n.name ASC, n.file ASC"
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

    let function_specific_filters = if node_types.iter().any(|nt| nt == &NodeType::Function) {
        format!("AND ({})", unique_functions_filters().join(" AND "))
    } else {
        "AND (n.body IS NOT NULL AND n.body <> '')".to_string()
    };

    let coverage_where = match coverage_filter {
        Some("tested") => "WHERE is_transitive_covered = true",
        Some("untested") => "WHERE is_transitive_covered = false",
        _ => "",
    };

    let query = format!(
        "// Step 1: Get all nodes in scope
         MATCH (n)
         WHERE ANY(label IN labels(n) WHERE label IN $node_types)
         {} {} {} {} {} {}
         WITH COLLECT(DISTINCT n) AS allNodes

         // Step 2: Find directly tested nodes with test counts
         OPTIONAL MATCH (test)-[:CALLS]->(direct)
         WHERE (test:UnitTest OR test:IntegrationTest OR test:E2etest)
         AND direct IN allNodes
         WITH allNodes, 
              COLLECT(DISTINCT direct) AS directlyTested,
              COLLECT({{node: direct, test: test}}) AS testPairs

         // Step 3: Find transitively covered nodes
         UNWIND CASE WHEN size(directlyTested) = 0 THEN [null] ELSE directlyTested END AS d
         OPTIONAL MATCH (d)-[:CALLS|NESTED_IN|HANDLER|OPERAND|RENDERS*1..{}]->(transitive)
         WHERE transitive IN allNodes
         WITH allNodes, directlyTested, testPairs,
              COLLECT(DISTINCT transitive) AS transitiveRaw

         // Step 4: Build transitive coverage set
         WITH allNodes, directlyTested, testPairs,
              directlyTested + [t IN transitiveRaw WHERE t IS NOT NULL AND NOT t IN directlyTested] AS transitivelyCovered

         // Step 5: Build result set with coverage info
         UNWIND allNodes AS n
         WITH n, directlyTested, transitivelyCovered, testPairs,
              n IN transitivelyCovered AS is_transitive_covered,
              n IN directlyTested AS is_direct_covered,
              size([pair IN testPairs WHERE pair.node = n]) AS direct_test_count

         // Step 6: Get caller counts
         OPTIONAL MATCH (caller)-[:CALLS]->(n)
         WITH n, is_transitive_covered, is_direct_covered, direct_test_count,
              count(DISTINCT caller) AS usage_count

         {}
         {}

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
         RETURN 
             size(all_items) AS total_count,
             [item IN all_items | item][$offset..($offset + $limit)] AS items",
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

fn unique_functions_filters() -> Vec<String> {
    vec![
        "n.body IS NOT NULL".to_string(),
        "n.body <> ''".to_string(),
        "(n.component IS NULL OR n.component <> 'true')".to_string(),
        "n.operand IS NULL".to_string(),
    ]
}
