use crate::lang::{
    helpers::{boltmap_insert_int, boltmap_insert_str},
    EdgeType, NodeData, NodeType,
};
use neo4rs::BoltMap;

pub fn count_nodes_edges_query() -> String {
    "MATCH (n) 
     WITH COUNT(n) as nodes
     MATCH ()-[r]->() 
     RETURN nodes, COUNT(r) as edges"
        .to_string()
}
pub fn graph_node_analysis_query() -> String {
    "MATCH (n) 
     RETURN n.node_key as node_key
     ORDER BY node_key"
        .to_string()
}
pub fn graph_edges_analysis_query() -> String {
    "MATCH (source)-[r]->(target) 
     RETURN source.node_key as source_key, type(r) as edge_type, target.node_key as target_key
     ORDER BY source_key, edge_type, target_key"
        .to_string()
}
pub fn count_edges_by_type_query(edge_type: &EdgeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());

    let query = "MATCH ()-[r]->() 
                WHERE type(r) = $edge_type 
                RETURN COUNT(r) as count";

    (query.to_string(), params)
}

pub fn find_resource_nodes_query(
    node_type: &NodeType,
    verb: &str,
    path: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "verb", verb);
    boltmap_insert_str(&mut params, "path", path);
    let query = format!(
        "MATCH (n:{})
         WHERE n.name CONTAINS $path AND 
               (n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb)
         RETURN n",
        node_type.to_string()
    );

    (query, params)
}

pub fn find_handlers_for_endpoint_query(endpoint: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "endpoint_name", &endpoint.name);
    boltmap_insert_str(&mut params, "endpoint_file", &endpoint.file);
    boltmap_insert_int(&mut params, "endpoint_start", endpoint.start as i64);

    let query = format!(
        "MATCH (endpoint:Endpoint {{name: $endpoint_name, file: $endpoint_file, start: $endpoint_start}})
        -[:HANDLER]->(handler)
        RETURN handler");

    (query, params)
}

pub fn check_direct_data_model_usage_query(
    function_name: &str,
    data_model: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "function_name", function_name);
    boltmap_insert_str(&mut params, "data_model", data_model);

    let query = format!(
        "MATCH (f:Function {{name: $function_name}})-[:CONTAINS]->(n:Datamodel)
         WHERE n.name CONTAINS $data_model
         RETURN COUNT(n) > 0 as exists"
    );

    (query, params)
}

pub fn find_functions_called_by_query(function: &NodeData) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "function_name", &function.name);
    boltmap_insert_str(&mut params, "function_file", &function.file);
    boltmap_insert_int(&mut params, "function_start", function.start as i64);

    let query = format!(
        "MATCH (source:Function {{name: $function_name, file: $function_file, start: $function_start}})
        -[:CALLS]->(target:Function)
        RETURN target");

    (query, params)
}

pub fn all_nodes_and_edges_query() -> (String, String) {
    let node_query = "MATCH (n) WHERE n.node_key IS NOT NULL RETURN DISTINCT n.node_key as key";
    let edge_query = "MATCH (s)-[r]->(t) WHERE s.node_key IS NOT NULL AND t.node_key IS NOT NULL RETURN DISTINCT s.node_key as source_key, t.node_key as target_key, type(r) as edge_type";

    (node_query.to_string(), edge_query.to_string())
}

pub fn class_inherits_query() -> String {
    "MATCH (c:Class)
    WHERE c.parent IS NOT NULL
    MATCH (parent:Class {name: c.parent})
    MERGE (parent)-[:PARENT_OF]->(c)"
        .to_string()
}
pub fn class_includes_query() -> String {
    "MATCH (c:Class)
    WHERE c.includes IS NOT NULL
    WITH c, split(c.includes, ',') AS modules
    UNWIND modules AS module
    MATCH (m:Class {name: trim(module)})
    MERGE (c)-[:IMPORTS]->(m)"
        .to_string()
}

pub fn endpoint_group_same_file_query() -> String {
    "MATCH (e:Endpoint) 
     WHERE e.file = $file 
       AND NOT e.name STARTS WITH $prefix
       AND NOT e.name CONTAINS '/:' 
       AND e.object = $object
     SET e.name = $prefix + e.name
     RETURN e.name as updated_name"
        .to_string()
}

pub fn endpoint_group_check_local_query() -> String {
    "MATCH (e:Endpoint)
     WHERE e.file = $file AND e.object = $object
     RETURN count(e) > 0 as is_local"
        .to_string()
}

pub fn endpoint_group_cross_file_query() -> String {
    "MATCH (e:Endpoint)
     WHERE e.file CONTAINS $resolved_source
       AND NOT e.name STARTS WITH $prefix
     SET e.name = $prefix + e.name
     RETURN e.name as updated_name"
        .to_string()
}

pub fn find_endpoint_query(name: &str, file: &str, verb: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "verb", verb.to_uppercase().as_str());
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_str(&mut params, "node_type", &NodeType::Endpoint.to_string());

    let query = "MATCH (n:Endpoint {name: $name, file: $file})
         WHERE n.verb IS NULL OR toUpper(n.verb) CONTAINS $verb
         RETURN n";

    (query.to_string(), params)
}
