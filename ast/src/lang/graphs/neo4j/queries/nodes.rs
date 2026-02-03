use super::edges::add_edge_query;
use crate::lang::{
    helpers::*,
    migration::{update_endpoint_name_query, update_endpoint_relationships_query},
    Edge, Node, NodeData, NodeType, Operand, TestFilters,
};
use crate::utils::create_node_key;
use neo4rs::{BoltMap, BoltType};
pub struct NodeQueryBuilder {
    node_type: NodeType,
    node_data: NodeData,
}

impl NodeQueryBuilder {
    pub fn new(node_type: &NodeType, node_data: &NodeData) -> Self {
        Self {
            node_type: node_type.clone(),
            node_data: node_data.clone(),
        }
    }

    pub fn build(&self) -> (String, BoltMap) {
        let mut properties: BoltMap = (&self.node_data).into();
        let ref_id = if std::env::var("TEST_REF_ID").is_ok() {
            "test_ref_id".to_string()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        boltmap_insert_str(&mut properties, "ref_id", &ref_id);

        let node_key = create_node_key(&Node::new(self.node_type.clone(), self.node_data.clone()));
        boltmap_insert_str(&mut properties, "node_key", &node_key);

        let token_count = calculate_token_count(&self.node_data.body).unwrap_or(0);
        boltmap_insert_int(&mut properties, "token_count", token_count);

        // Add Data_Bank property during node creation (fixes real-time streaming)
        if !self.node_data.name.is_empty() {
            boltmap_insert_str(&mut properties, "Data_Bank", &self.node_data.name);
        }

        // Add default namespace during node creation (fixes real-time streaming)
        boltmap_insert_str(&mut properties, "namespace", "default");

        // println!("[NodeQueryBuilder] node_key: {}", node_key);

        let query = format!(
            "MERGE (node:{}:{} {{node_key: $node_key}})
         ON CREATE SET node += $properties, node.date_added_to_graph = $now
         ON MATCH SET node += $properties, node.date_added_to_graph = $now
         Return node",
            self.node_type.to_string(),
            DATA_BANK,
        );

        (query, properties)
    }

    pub fn build_stream(&self) -> (String, BoltMap) {
        let mut properties: BoltMap = (&self.node_data).into();
        let ref_id = if std::env::var("TEST_REF_ID").is_ok() {
            "test_ref_id".to_string()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        boltmap_insert_str(&mut properties, "ref_id", &ref_id);

        let node_key = create_node_key(&Node::new(self.node_type.clone(), self.node_data.clone()));
        boltmap_insert_str(&mut properties, "node_key", &node_key);

        let token_count = calculate_token_count(&self.node_data.body).unwrap_or(0);
        boltmap_insert_int(&mut properties, "token_count", token_count);

        // Add Data_Bank property during node creation (fixes real-time streaming)
        if !self.node_data.name.is_empty() {
            boltmap_insert_str(&mut properties, "Data_Bank", &self.node_data.name);
        }

        // Add default namespace during node creation (fixes real-time streaming)
        boltmap_insert_str(&mut properties, "namespace", "default");

        // println!("[NodeQueryBuilder] node_key: {}", node_key);

        let query = format!(
            "MERGE (node:{}:{} {{node_key: $node_key}})
         ON CREATE SET node += $properties, node.date_added_to_graph = $now
         Return node",
            self.node_type.to_string(),
            DATA_BANK,
        );

        (query, properties)
    }
}

pub fn add_node_query(node_type: &NodeType, node_data: &NodeData) -> (String, BoltMap) {
    NodeQueryBuilder::new(node_type, node_data).build()
}

pub fn add_node_query_stream(node_type: &NodeType, node_data: &NodeData) -> (String, BoltMap) {
    NodeQueryBuilder::new(node_type, node_data).build_stream()
}

pub fn add_node_with_parent_query(
    node_type: &NodeType,
    node_data: &NodeData,
    parent_type: &NodeType,
    parent_file: &str,
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(node_type, node_data));

    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_name", &node_data.name);
    boltmap_insert_str(&mut params, "node_file", &node_data.file);
    boltmap_insert_int(&mut params, "node_start", node_data.start as i64);
    boltmap_insert_str(&mut params, "parent_file", parent_file);

    let query_str = format!(
        "MATCH (parent:{} {{file: $parent_file}})
         MATCH (node:{} {{name: $node_name, file: $node_file, start: $node_start}})
         MERGE (parent)-[:CONTAINS]->(node)",
        parent_type.to_string(),
        node_type.to_string()
    );

    queries.push((query_str, params));
    queries
}
pub fn add_functions_query(
    function_node: &NodeData,
    method_of: Option<&Operand>,
    reqs: &[NodeData],
    dms: &[Edge],
    trait_operand: Option<&Edge>,
    return_types: &[Edge],
    nested_in: &[Edge],
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Function, function_node));

    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "function_name", &function_node.name);
    boltmap_insert_str(&mut params, "function_file", &function_node.file);
    boltmap_insert_int(&mut params, "function_start", function_node.start as i64);

    let query_str = format!(
        "MATCH (function:Function {{name: $function_name, file: $function_file, start: $function_start}}),
               (file:File {{file: $function_file}})
         MERGE (file)-[:CONTAINS]->(function)"
    );
    queries.push((query_str, params));

    if let Some(operand) = method_of {
        let edge = (*operand).clone().into();
        queries.push(add_edge_query(&edge));
    }

    if let Some(edge) = trait_operand {
        queries.push(add_edge_query(edge));
    }

    for edge in return_types {
        queries.push(add_edge_query(edge));
    }

    for req in reqs {
        queries.push(add_node_query(&NodeType::Request, req));

        let mut params = BoltMap::new();
        boltmap_insert_str(&mut params, "function_name", &function_node.name);
        boltmap_insert_str(&mut params, "function_file", &function_node.file);
        boltmap_insert_int(&mut params, "function_start", function_node.start as i64);
        boltmap_insert_str(&mut params, "req_name", &req.name);
        boltmap_insert_str(&mut params, "req_file", &req.file);
        boltmap_insert_int(&mut params, "req_start", req.start as i64);
        let query_str = format!(
            "MATCH (function:Function {{name: $function_name, file: $function_file, start: $function_start}}),
                   (request:Request {{name: $req_name, file: $req_file, start: $req_start}})
             MERGE (function)-[:CALLS]->(request)"
        );
        queries.push((query_str, params));
    }

    for dm_edge in dms {
        queries.push(add_edge_query(dm_edge));
    }
    for ne_edge in nested_in {
        queries.push(add_edge_query(ne_edge));
    }
    queries
}

pub fn add_page_query(page_data: &NodeData, edge_opt: &Option<Edge>) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    queries.push(add_node_query(&NodeType::Page, page_data));

    if let Some(edge) = edge_opt {
        queries.push(add_edge_query(edge));
    }

    queries
}

pub fn add_pages_query(pages: &[(NodeData, Vec<Edge>)]) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (page_data, edges) in pages {
        queries.push(add_node_query(&NodeType::Page, page_data));

        for edge in edges {
            queries.push(add_edge_query(edge));
        }
    }

    queries
}
pub fn add_instance_of_query(instance: &NodeData, class_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "instance_name", &instance.name);
    boltmap_insert_str(&mut params, "instance_file", &instance.file);
    boltmap_insert_int(&mut params, "instance_start", instance.start as i64);
    boltmap_insert_str(&mut params, "class_name", class_name);

    let query = "MATCH (instance:Instance {name: $instance_name, file: $instance_file, start: $instance_start}), 
                       (class:Class {name: $class_name}) 
                 MERGE (instance)-[:OF]->(class)";

    (query.to_string(), params)
}

pub fn add_endpoints_query(endpoints: &[(NodeData, Option<Edge>)]) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (endpoint_data, handler_edge) in endpoints {
        queries.push(add_node_query(&NodeType::Endpoint, endpoint_data));

        if let Some(edge) = handler_edge {
            queries.push(add_edge_query(edge));
        }
    }

    queries
}

pub fn add_instance_contains_query(instance: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "instance_name", &instance.name);
    boltmap_insert_str(&mut params, "instance_file", &instance.file);
    boltmap_insert_int(&mut params, "instance_start", instance.start as i64);

    let query = "MATCH (file:File {file: $instance_file}),
                       (instance:Instance {name: $instance_name, file: $instance_file, start: $instance_start})
                 MERGE (file)-[:CONTAINS]->(instance)";

    (query.to_string(), params)
}

pub fn find_nodes_by_type_query(node_type: &NodeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());

    let query = format!(
        "MATCH (n:{}) 
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}
pub fn find_nodes_by_name_query(node_type: &NodeType, name: &str, root: &str) -> (String, BoltMap) {
    let mut param = BoltMap::new();
    param
        .value
        .insert("name".into(), BoltType::String(name.into()));
    param
        .value
        .insert("root".into(), BoltType::String(root.into()));

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name
                       AND n.file STARTS WITH $root
                       RETURN n",
        node_type.to_string()
    );

    (query, param)
}

pub fn find_node_by_name_file_query(
    node_type: &NodeType,
    name: &str,
    file: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "file", file);

    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.name = $name AND n.file = $file 
                       RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_nodes_by_file_pattern_query(
    node_type: &NodeType,
    file_pattern: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "file_pattern", file_pattern);
    let query = format!(
        "MATCH (n:{}) 
                       WHERE n.file CONTAINS $file_pattern 
                       RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_nodes_by_name_contains_query(
    node_type: &NodeType,
    name_part: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name_part", name_part);

    let query = format!(
        "MATCH (n:{}) 
         WHERE n.name CONTAINS $name_part 
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}
pub fn find_nodes_in_range_query(node_type: &NodeType, file: &str, row: u32) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_int(&mut params, "row", row as i64);

    let query = format!(
        "MATCH (n:$node_type)
         WHERE n.file = $file AND 
               toInteger(n.start) <= toInteger($row) AND 
               toInteger(n.end) >= toInteger($row)
         RETURN n"
    );

    (query, params)
}
pub fn find_node_at_query(node_type: &NodeType, file: &str, line: u32) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_int(&mut params, "line", line as i64);

    let query = format!(
        "MATCH (n:{}) 
         WHERE n.file = $file AND 
               toInteger(n.start) <= toInteger($line) AND 
               toInteger(n.end) >= toInteger($line)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn all_node_keys_query() -> String {
    "MATCH (n) WHERE n.node_key IS NOT NULL RETURN n.node_key as node_key".to_string()
}

pub fn filter_out_nodes_without_children_query(
    parent_type: NodeType,
    child_type: NodeType,
    _child_meta_key: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "parent_type", &parent_type.to_string());
    boltmap_insert_str(&mut params, "child_type", &child_type.to_string());

    let query = format!(
        "MATCH (parent:{})
        WHERE NOT EXISTS {{
            MATCH (parent)-[:OPERAND]->(child:{})
        }}
        AND NOT EXISTS {{
            MATCH (instance:Instance)-[:OF]->(parent)
        }}
        DETACH DELETE parent",
        parent_type.to_string(),
        child_type.to_string()
    );

    (query, params)
}

pub fn remove_node_query(node_type: NodeType, node_data: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", &node_data.name);
    boltmap_insert_str(&mut params, "file", &node_data.file);
    boltmap_insert_int(&mut params, "start", node_data.start as i64);

    let verb_clause = if node_type == NodeType::Endpoint {
        if let Some(verb) = node_data.meta.get("verb") {
            boltmap_insert_str(&mut params, "verb", verb);
            "AND n.verb = $verb"
        } else {
            ""
        }
    } else {
        ""
    };

    let query = format!(
        "MATCH (n:{} {{name: $name, file: $file, start: $start}})
         WHERE true {}
         DETACH DELETE n",
        node_type.to_string(),
        verb_clause
    );

    (query, params)
}

pub fn find_group_function_query(group_function_name: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "group_function_name", group_function_name);

    let query = "MATCH (n:Function) 
                 WHERE n.name = $group_function_name 
                 RETURN n";

    (query.to_string(), params)
}

pub fn find_top_level_functions_query() -> (String, BoltMap) {
    let query = format!(
        "MATCH (n:Function)
        WHERE {}
        RETURN n
    ",
        unique_functions_filters().join(" AND ")
    )
    .to_string();
    (query, BoltMap::new())
}

pub fn process_endpoint_groups_queries(
    groups_with_endpoints: &[(NodeData, Vec<NodeData>)],
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (group, endpoints) in groups_with_endpoints {
        for endpoint in endpoints {
            let new_name = format!("{}{}", group.name, endpoint.name);
            let mut new_node_data = endpoint.clone();
            new_node_data.name = new_name.clone();
            let new_key = create_node_key(&Node::new(NodeType::Endpoint, new_node_data));

            queries.push(update_endpoint_name_query(
                &endpoint.name,
                &endpoint.file,
                &new_name,
                &new_key,
            ));

            queries.push(update_endpoint_relationships_query(
                &endpoint.name,
                &endpoint.file,
                &new_name,
            ));
        }
    }

    queries
}

pub fn get_muted_nodes_for_files_query(files: &[String]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let files_list = files
        .iter()
        .map(|f| BoltType::String(f.clone().into()))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "files", files_list);

    let query = "MATCH (n) 
                 WHERE (n.file IN $files OR any(f IN $files WHERE n.file ENDS WITH f))
                 AND (n.is_muted = true OR n.is_muted = 'true')
                 WITH n, [label IN labels(n) WHERE label IN ['Function', 'Class', 'DataModel', 'Endpoint', 'Request', 'File', 'Directory', 'Repository', 'Language', 'Library', 'Import', 'Instance', 'Page', 'Var', 'UnitTest', 'IntegrationTest', 'E2eTest', 'Trait']][0] as node_type
                 WHERE node_type IS NOT NULL
                 RETURN node_type, n.name as name, n.file as file".to_string();

    (query, params)
}

pub fn restore_muted_status_query(identifiers: &[MutedNodeIdentifier]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let identifier_maps: Vec<BoltType> = identifiers
        .iter()
        .map(|ident| {
            let mut map = BoltMap::new();
            boltmap_insert_str(&mut map, "node_type", &ident.node_type.to_string());
            boltmap_insert_str(&mut map, "name", &ident.name);
            boltmap_insert_str(&mut map, "file", &ident.file);
            BoltType::Map(map)
        })
        .collect();
    boltmap_insert_list(&mut params, "identifiers", identifier_maps);

    let query = "UNWIND $identifiers as ident
                 MATCH (n)
                 WHERE ident.node_type IN labels(n) 
                 AND n.name = ident.name 
                 AND n.file = ident.file
                 SET n.is_muted = true
                 RETURN count(n) as restored_count"
        .to_string();

    (query, params)
}

pub fn query_nodes_with_count(
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

    let order_clause = if body_length {
        "ORDER BY size(n.body) DESC, n.name ASC, n.file ASC"
    } else if line_count {
        "ORDER BY (n.end - n.start) DESC, n.name ASC, n.file ASC"
    } else if sort_by_test_count {
        "ORDER BY test_count DESC, n.name ASC, n.file ASC"
    } else {
        "ORDER BY n.name ASC, n.file ASC"
    };

    let (test_match_clauses, test_count_expr) = if let Some(filters) = &test_filters {
        let has_filters = !filters.unit_regexes.is_empty()
            || !filters.integration_regexes.is_empty()
            || !filters.e2e_regexes.is_empty();

        if has_filters {
            let mut clauses = Vec::new();
            let mut count_parts = Vec::new();

            if !filters.unit_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .unit_regexes
                    .iter()
                    .map(|r| format!("ut.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (ut:UnitTest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT ut)");
            }

            if !filters.integration_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .integration_regexes
                    .iter()
                    .map(|r| format!("it.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (it:IntegrationTest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT it)");
            }

            if !filters.e2e_regexes.is_empty() {
                let regex_conditions: Vec<String> = filters
                    .e2e_regexes
                    .iter()
                    .map(|r| format!("et.file =~ '{}'", r))
                    .collect();
                clauses.push(format!(
                    "OPTIONAL MATCH (et:E2etest)-[:CALLS]->(n) WHERE {}",
                    regex_conditions.join(" OR ")
                ));
                count_parts.push("COUNT(DISTINCT et)");
            }

            let test_count = if count_parts.is_empty() {
                "0".to_string()
            } else {
                count_parts.join(" + ")
            };

            (clauses.join("\n         "), test_count)
        } else {
            (
                "OPTIONAL MATCH (test)-[:CALLS]->(n) WHERE test:UnitTest OR test:IntegrationTest OR test:E2etest".to_string(),
                "count(DISTINCT test)".to_string(),
            )
        }
    } else {
        (
            "OPTIONAL MATCH (test)-[:CALLS]->(n) WHERE test:UnitTest OR test:IntegrationTest OR test:E2etest".to_string(),
            "count(DISTINCT test)".to_string(),
        )
    };

    let coverage_where = match coverage_filter {
        Some("tested") => "WHERE test_count > 0 OR n.indirect_test IS NOT NULL",
        Some("untested") => "WHERE test_count = 0 AND n.indirect_test IS NULL",
        _ => "",
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

    let has_function_type = node_types.iter().any(|nt| nt == &NodeType::Function);

    let function_specific_filters = if has_function_type {
        format!("AND ({})", unique_functions_filters().join(" AND "))
    } else {
        "AND (n.body IS NOT NULL AND n.body <> '')".to_string()
    };

    let query = format!(
        "MATCH (n)
         WHERE ANY(label IN labels(n) WHERE label IN $node_types)
         {} {} {} {} {} {}
         {}
         WITH n, {} AS test_count
         {} 
         OPTIONAL MATCH (caller)-[:CALLS]->(n)
         WITH n, test_count, count(DISTINCT caller) AS usage_count, (test_count > 0 OR n.indirect_test IS NOT NULL) AS is_covered
         {}
         WITH collect({{
             node: n,
             usage_count: usage_count,
             is_covered: is_covered,
             test_count: test_count,
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
        test_match_clauses,
        test_count_expr,
        coverage_where,
        order_clause
    );

    (query, params)
}
