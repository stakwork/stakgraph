use crate::lang::graphs::helpers::{
    boltmap_insert_float, boltmap_insert_int, boltmap_insert_list, boltmap_insert_str,
};
use lsp::Language;
use neo4rs::{BoltFloat, BoltMap, BoltString, BoltType};
use std::str::FromStr;

pub fn data_bank_bodies_query_no_embeddings(
    do_files: bool,
    skip: usize,
    limit: usize,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_int(&mut params, "skip", skip as i64);
    boltmap_insert_int(&mut params, "limit", limit as i64);
    boltmap_insert_str(
        &mut params,
        "do_files",
        if do_files { "true" } else { "false" },
    );
    let query = r#"
        MATCH (n:Data_Bank)
        WHERE n.embeddings IS NULL
          AND (($do_files = 'true') OR NOT n:File)
        RETURN n.node_key as node_key, n.body as body
        SKIP toInteger($skip) LIMIT toInteger($limit)
    "#
    .to_string();
    (query, params)
}
pub fn update_embedding_query(node_key: &str, embedding: &[f32]) -> (String, BoltMap) {
    let mut params = BoltMap::new();
    boltmap_insert_str(&mut params, "node_key", node_key);
    let emb_list = embedding
        .iter()
        .map(|&v| BoltType::Float(BoltFloat { value: v as f64 }))
        .collect::<Vec<_>>();
    boltmap_insert_list(&mut params, "embeddings", emb_list);
    let query = r#"
        MATCH (n:Data_Bank {node_key: $node_key})
        SET n.embeddings = $embeddings
    "#
    .to_string();
    (query, params)
}
pub fn vector_search_query(
    embedding: &[f32],
    limit: usize,
    node_types: Vec<String>,
    similarity_threshold: f32,
    language: Option<String>,
) -> (String, BoltMap) {
    let mut params = BoltMap::new();

    let emb_list = embedding
        .iter()
        .map(|&v| BoltType::Float(BoltFloat { value: v as f64 }))
        .collect::<Vec<_>>();

    boltmap_insert_list(&mut params, "embeddings", emb_list.clone());

    boltmap_insert_int(&mut params, "limit", limit as i64);

    boltmap_insert_float(
        &mut params,
        "similarityThreshold",
        similarity_threshold as f64,
    );

    let node_types_list = node_types
        .into_iter()
        .map(|s| BoltType::String(BoltString::from(s)))
        .collect::<Vec<_>>();

    boltmap_insert_list(&mut params, "node_types", node_types_list);

    let ext_list = if let Some(lang_str) = language.as_ref() {
        if let Ok(lang) = Language::from_str(lang_str) {
            lang.exts()
                .iter()
                .map(|&s| BoltType::String(s.into()))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    boltmap_insert_list(&mut params, "extensions", ext_list);

    let query = r#"
        MATCH (node)
        WHERE
          CASE
            WHEN $node_types IS NULL OR size($node_types) = 0 THEN true
            ELSE ANY(label IN labels(node) WHERE label IN $node_types)
          END
          AND node.embeddings IS NOT NULL
          AND
          CASE
            WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
            ELSE node.file IS NOT NULL AND ANY(ext IN $extensions WHERE node.file ENDS WITH ext)
          END
        WITH node, gds.similarity.cosine(node.embeddings, $embeddings) AS score
        WHERE score >= $similarityThreshold
        RETURN node, score
        ORDER BY score DESC
        LIMIT toInteger($limit)
    "#
    .to_string();

    (query, params)
}
