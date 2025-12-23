use neo4rs::{BoltMap, BoltType};
use lazy_static::lazy_static;
use shared::Result;
use crate::lang::{NodeData, NodeType};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};


pub const DATA_BANK: &str = "Data_Bank";
pub const BATCH_SIZE: usize = 4096;

lazy_static! {
    static ref TOKENIZER: CoreBPE = get_bpe_from_model("gpt-4").unwrap();
}

#[derive(Debug, Clone)]
pub struct MutedNodeIdentifier {
    pub node_type: NodeType,
    pub name: String,
    pub file: String,
}



pub fn boltmap_insert_str(map: &mut BoltMap, key: &str, value: &str) {
    map.value.insert(key.into(), BoltType::String(value.into()));
}
pub fn boltmap_insert_map(map: &mut BoltMap, key: &str, value: BoltMap) {
    map.value.insert(key.into(), BoltType::Map(value));
}
pub fn boltmap_insert_list_of_maps(map: &mut BoltMap, key: &str, value: Vec<BoltMap>) {
    let list = neo4rs::BoltList {
        value: value.into_iter().map(|m| BoltType::Map(m)).collect(),
    };
    map.value.insert(key.into(), BoltType::List(list));
}

pub fn boltmap_insert_list(map: &mut BoltMap, key: &str, value: Vec<BoltType>) {
    let list = neo4rs::BoltList { value };
    map.value.insert(key.into(), BoltType::List(list));
}
pub fn boltmap_insert_float(map: &mut BoltMap, key: &str, value: f64) {
    map.value.insert(
        key.into(),
        BoltType::Float(neo4rs::BoltFloat {
            value: value as f64,
        }),
    );
}
pub fn boltmap_insert_int(map: &mut BoltMap, key: &str, value: i64) {
    map.value
        .insert(key.into(), BoltType::Integer(value.into()));
}
pub fn boltmap_to_bolttype_map(bolt_map: BoltMap) -> BoltType {
    BoltType::Map(bolt_map)
}
pub fn boltmap_insert_bool(map: &mut BoltMap, key: &str, value: bool) {
    map.value.insert(key.into(), neo4rs::BoltType::Boolean(neo4rs::BoltBoolean { value }));
}


pub fn calculate_token_count(body: &str) -> Result<i64> {
    let bpe = &TOKENIZER;
    let token_count = bpe.encode_with_special_tokens(body).len() as i64;
    Ok(token_count)
}

pub fn unique_functions_filters() -> Vec<String> {
    vec![
        "NOT (n)-[:NESTED_IN]->(:Function)".to_string(),
        "(n.body IS NOT NULL AND n.body <> '')".to_string(),
        "NOT (:Endpoint)-[:HANDLER]->(n)".to_string(),
        "(n.component IS NULL OR n.component <> 'true')".to_string(),
        "n.operand IS NULL".to_string(),
        "EXISTS(()-[:CALLS]->(:Function))".to_string(),
    ]
}

pub fn extract_ref_id(node_data: &NodeData) -> String {
    node_data
        .meta
        .get("ref_id")
        .cloned()
        .unwrap_or_else(|| "placeholder".to_string())
}


pub fn set_node_muted_query(node_type: &NodeType, name: &str, file: &str, is_muted: bool) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "file", file);
    boltmap_insert_bool(&mut params, "is_muted", is_muted);
    
    let query = "MATCH (n {name: $name, file: $file}) 
                 WHERE $node_type IN labels(n) 
                 SET n.is_muted = $is_muted 
                 RETURN count(n) as updated_count";
    (query.to_string(), params)
}

pub fn check_node_muted_query(node_type: &NodeType, name: &str, file: &str) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_type", &node_type.to_string());
    boltmap_insert_str(&mut params, "name", name);
    boltmap_insert_str(&mut params, "file", file);
    
    let query = "MATCH (n {name: $name, file: $file}) 
                 WHERE $node_type IN labels(n) 
                 RETURN n.is_muted as is_muted";
    (query.to_string(), params)
}