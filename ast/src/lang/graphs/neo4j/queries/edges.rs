use crate::lang::graphs::{
    queries::add_node_query, Calls, Edge, EdgeType, Graph, Node, NodeData, NodeRef, NodeType,
};
use crate::lang::helpers::{boltmap_insert_list_of_maps, boltmap_insert_str};
use crate::utils::create_node_key_from_ref;
use neo4rs::BoltMap;

pub struct EdgeQueryBuilder {
    edge: Edge,
}

impl EdgeQueryBuilder {
    pub fn new(edge: &Edge) -> Self {
        Self { edge: edge.clone() }
    }

    pub fn build(&self) -> (String, BoltMap) {
        let mut params = BoltMap::new();

        let rel_type = self.edge.edge.to_string();

        let source_type = self.edge.source.node_type.to_string();
        let source_key = create_node_key_from_ref(&self.edge.source);
        boltmap_insert_str(&mut params, "source_key", &source_key);

        let target_type = self.edge.target.node_type.to_string();
        let target_key = create_node_key_from_ref(&self.edge.target);
        boltmap_insert_str(&mut params, "target_key", &target_key);
        boltmap_insert_str(&mut params, "ref_id", &self.edge.ref_id);

        // println!(
        //     "[EdgeQueryBuilder] source_key: {}, target_key: {}",
        //     source_key, target_key
        // );

        let query = format!(
            "MATCH (source:{} {{node_key: $source_key}}),
                 (target:{} {{node_key: $target_key}})
            MERGE (source)-[r:{}]->(target)
            SET r.ref_id = $ref_id
            RETURN r",
            source_type, target_type, rel_type
        );
        (query, params)
    }

    pub fn build_stream(&self) -> (String, BoltMap) {
        let mut params = BoltMap::new();

        let rel_type = self.edge.edge.to_string();

        let source_type = self.edge.source.node_type.to_string();
        let source_key = create_node_key_from_ref(&self.edge.source);
        boltmap_insert_str(&mut params, "source_key", &source_key);

        let target_type = self.edge.target.node_type.to_string();
        let target_key = create_node_key_from_ref(&self.edge.target);
        boltmap_insert_str(&mut params, "target_key", &target_key);
        boltmap_insert_str(&mut params, "ref_id", &self.edge.ref_id);

        let query = format!(
            "MATCH (source:{} {{node_key: $source_key}}),
                 (target:{} {{node_key: $target_key}})
            MERGE (source)-[r:{}]->(target)
            SET r.ref_id = $ref_id",
            source_type, target_type, rel_type
        );
        (query, params)
    }
}

pub fn add_edge_query(edge: &Edge) -> (String, BoltMap) {
    EdgeQueryBuilder::new(edge).build()
}

pub fn add_edge_query_stream(edge: &Edge) -> (String, BoltMap) {
    EdgeQueryBuilder::new(edge).build_stream()
}

pub fn add_calls_query(
    funcs: &[(Calls, Option<NodeData>, Option<NodeData>)],
    tests: &[(Calls, Option<NodeData>, Option<NodeData>)],
    int_tests: &[Edge],
    extras: &[Edge],
    lang: &crate::lang::Lang,
    graph: &impl Graph,
) -> Vec<(String, BoltMap)> {
    let mut queries = Vec::new();

    for (calls, ext_func, class_call) in funcs {
        if let Some(class_call) = class_call {
            let edge = Edge::new(
                EdgeType::Calls,
                NodeRef::from(calls.source.clone(), NodeType::Function),
                NodeRef::from(class_call.into(), NodeType::Class),
            );
            queries.push(add_edge_query(&edge));
        }

        if calls.target.is_empty() {
            continue;
        }
        if let Some(ext_nd) = ext_func {
            queries.push(add_node_query(&NodeType::Function, ext_nd));
            let edge = Edge::uses(calls.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge: Edge = calls.clone().into();
            queries.push(add_edge_query(&edge));
        }
    }

    for (test_call, ext_func, class_call) in tests {
        if let Some(ext_nd) = ext_func {
            queries.push(add_node_query(&NodeType::Function, ext_nd));
            let edge = Edge::uses(test_call.source.clone(), ext_nd);
            queries.push(add_edge_query(&edge));
        } else {
            let edge = Edge::from_test_call(test_call, lang, graph);
            queries.push(add_edge_query(&edge));
        }

        if let Some(class_nd) = class_call {
            let edge = Edge::from_test_class_call(test_call, class_nd, lang, graph);
            queries.push(add_edge_query(&edge));

            queries.push(add_node_query(&NodeType::Class, class_nd));
        }
    }

    for edge in int_tests {
        queries.push(add_edge_query(edge));
    }
    for edge in extras {
        queries.push(add_edge_query(edge));
    }

    queries
}

pub fn build_batch_edge_queries<I>(edges: I, batch_size: usize) -> Vec<(String, BoltMap)>
where
    I: Iterator<Item = (String, String, EdgeType, String)>,
{
    use itertools::Itertools;
    use std::collections::HashMap;

    // Group edges by type
    let edges_by_type: HashMap<EdgeType, Vec<(String, String, String)>> = edges
        .map(|(source, target, edge_type, ref_id)| (edge_type, (source, target, ref_id)))
        .into_group_map();

    // Create batched queries for each edge type
    edges_by_type
        .into_iter()
        .flat_map(|(edge_type, type_edges)| {
            // Batch the edges for this type
            let chunks: Vec<Vec<(String, String, String)>> = type_edges
                .into_iter()
                .chunks(batch_size)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            chunks
                .into_iter()
                .map(|chunk| {
                    let edges_data: Vec<BoltMap> = chunk
                        .into_iter()
                        .map(|(source, target, ref_id)| {
                            let mut edge_map = BoltMap::new();
                            boltmap_insert_str(&mut edge_map, "source", &source);
                            boltmap_insert_str(&mut edge_map, "target", &target);
                            boltmap_insert_str(&mut edge_map, "ref_id", &ref_id);
                            edge_map
                        })
                        .collect();

                    let mut params = BoltMap::new();
                    boltmap_insert_list_of_maps(&mut params, "edges", edges_data);

                    let query = format!(
                        "UNWIND $edges AS edge
                         MATCH (source:Data_Bank {{node_key: edge.source}}), (target:Data_Bank {{node_key: edge.target}})
                         MERGE (source)-[r:{}]->(target)
                         SET r.ref_id = edge.ref_id
                         RETURN count(r)",
                        edge_type.to_string()
                    );

                    (query, params)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn build_batch_edge_queries_stream<I>(edges: I, batch_size: usize) -> Vec<(String, BoltMap)>
where
    I: Iterator<Item = (String, String, EdgeType, String)>,
{
    use itertools::Itertools;
    use std::collections::HashMap;

    let edges_by_type: HashMap<EdgeType, Vec<(String, String, String)>> = edges
        .map(|(source, target, edge_type, ref_id)| (edge_type, (source, target, ref_id)))
        .into_group_map();

    edges_by_type
        .into_iter()
        .flat_map(|(edge_type, type_edges)| {
            let chunks: Vec<Vec<(String, String, String)>> = type_edges
                .into_iter()
                .chunks(batch_size)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect();

            chunks
                .into_iter()
                .map(|chunk| {
                    let edges_data: Vec<BoltMap> = chunk
                        .into_iter()
                        .map(|(source, target, ref_id)| {
                            let mut edge_map = BoltMap::new();
                            boltmap_insert_str(&mut edge_map, "source", &source);
                            boltmap_insert_str(&mut edge_map, "target", &target);
                            boltmap_insert_str(&mut edge_map, "ref_id", &ref_id);
                            edge_map
                        })
                        .collect();

                    let mut params = BoltMap::new();
                    boltmap_insert_list_of_maps(&mut params, "edges", edges_data);

                    let query = format!(
                        "UNWIND $edges AS edge
                         MATCH (source:Data_Bank {{node_key: edge.source}}), (target:Data_Bank {{node_key: edge.target}})
                         MERGE (source)-[r:{}]->(target)
                         SET r.ref_id = edge.ref_id",
                        edge_type.to_string()
                    );

                    (query, params)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn find_source_edge_by_name_and_file_query(
    edge_type: &EdgeType,
    target_name: &str,
    target_file: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    boltmap_insert_str(&mut params, "target_name", target_name);
    boltmap_insert_str(&mut params, "target_file", target_file);
    let query = format!(
        "MATCH (source)-[r:{}]->(target {{name: $target_name, file: $target_file}})
         RETURN source.name as name, source.file as file, source.start as start, source.verb as verb
         LIMIT 1",
        edge_type.to_string()
    );
    (query, params)
}

pub fn find_nodes_with_edge_type_query(
    source_type: &NodeType,
    target_type: &NodeType,
    edge_type: &EdgeType,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "source_type", &source_type.to_string());
    boltmap_insert_str(&mut params, "target_type", &target_type.to_string());
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    let query = format!(
        "MATCH (source:{})-[r:{}]->(target:{})
         RETURN source.name as source_name, source.file as source_file, source.start as source_start, \
                target.name as target_name, target.file as target_file, target.start as target_start",
        source_type.to_string(),
        edge_type.to_string(),
        target_type.to_string()
    );

    (query, params)
}

pub fn find_dynamic_edges_for_file_query(file: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "file", file);

    let static_types = vec![
        "Repository",
        "Package",
        "Language",
        "Directory",
        "File",
        "Import",
        "Library",
        "Class",
        "Trait",
        "Instance",
        "Function",
        "Endpoint",
        "Request",
        "Datamodel",
        "Feature",
        "Page",
        "Var",
        "UnitTest",
        "IntegrationTest",
        "E2etest",
    ];

    let static_labels = static_types
        .iter()
        .map(|t| format!("source:{}", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let query = format!(
        "MATCH (source)-[r]->(target)
         WHERE target.file ENDS WITH $file 
         AND NOT ({})
         RETURN source.ref_id as source_ref_id, type(r) as edge_type, 
                target.name as target_name, target.file as target_file, labels(target)[0] as target_type",
        static_labels
    );

    (query, params)
}

pub fn find_all_dynamic_edges_query() -> (String, BoltMap) {
    let params = BoltMap::new();

    let static_types = vec![
        "Repository",
        "Package",
        "Language",
        "Directory",
        "File",
        "Import",
        "Library",
        "Class",
        "Trait",
        "Instance",
        "Function",
        "Endpoint",
        "Request",
        "Datamodel",
        "Feature",
        "Page",
        "Var",
        "UnitTest",
        "IntegrationTest",
        "E2etest",
    ];

    let static_labels = static_types
        .iter()
        .map(|t| format!("source:{}", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let query = format!(
        "MATCH (source)-[r]->(target)
         WHERE NOT ({})
         RETURN source.ref_id as source_ref_id, type(r) as edge_type, 
                target.name as target_name, target.file as target_file, labels(target)[0] as target_type",
        static_labels
    );

    (query, params)
}

pub fn restore_dynamic_edge_query(
    source_ref_id: &str,
    edge_type: &str,
    target_name: &str,
    target_file: &str,
    target_type: &str,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "source_ref_id", source_ref_id);
    boltmap_insert_str(&mut params, "edge_type", edge_type);
    boltmap_insert_str(&mut params, "target_name", target_name);
    boltmap_insert_str(&mut params, "target_file", target_file);

    let query = format!(
        "MATCH (source {{ref_id: $source_ref_id}})
         MATCH (target:{} {{name: $target_name, file: $target_file}})
         MERGE (source)-[r:{}]->(target)
         RETURN r",
        target_type, edge_type
    );

    (query, params)
}
pub fn all_edge_triples_query() -> String {
    "MATCH (s)-[e]->(t) RETURN s.node_key as s_key, type(e) as edge_type, t.node_key as t_key"
        .to_string()
}

pub fn has_edge_query(source: &Node, target: &Node, edge_type: &EdgeType) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    let source_type = &source.node_type;
    let target_type = &target.node_type;

    boltmap_insert_str(&mut params, "source_type", &source.node_type.to_string());
    boltmap_insert_str(&mut params, "target_type", &target.node_type.to_string());
    boltmap_insert_str(&mut params, "edge_type", &edge_type.to_string());
    boltmap_insert_str(&mut params, "source_name", &source.node_data.name);
    boltmap_insert_str(&mut params, "source_file", &source.node_data.file);
    boltmap_insert_str(&mut params, "target_name", &target.node_data.name);
    boltmap_insert_str(&mut params, "target_file", &target.node_data.file);

    let query = format!(
        "MATCH (source:{})-[r:{}]->(target:{})
         WHERE source.name = $source_name AND source.file = $source_file
           AND target.name = $target_name AND target.file = $target_file
         RETURN COUNT(r) > 0 as exists",
        source_type.to_string(),
        edge_type.to_string(),
        target_type.to_string()
    );

    (query, params)
}
