use ast::lang::asg::NodeData;
use ast::lang::NodeType;
use shared::Result;
use std::str::FromStr;

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

pub async fn run_coverage(repo_root: &Path, repo_url: &str) {
    if let Some(repo_path_str) = repo_root.to_str() {
        let commit = lsp::git::get_commit_hash(repo_path_str).await.ok().unwrap_or_default();
        match shared::codecov::run(repo_path_str, repo_url, &commit) {
            Ok(report) => tracing::info!("coverage run ok repo={} commit={} langs={} errors={}", repo_url, commit, report.languages.len(), report.errors.len()),
            Err(e) => tracing::warn!("coverage run failed repo={} commit={} err={}", repo_url, commit, e),
        }
    }
}

pub fn covered_line_ranges(repo_root: &Path) -> HashMap<String, Vec<(u32,u32)>> {
    fn normalize_variants(original: &str, repo_root: &Path) -> Vec<String> {
        let mut out: HashSet<String> = HashSet::new();
        let orig = original.to_string();
        out.insert(orig.clone());

        if let Some(stripped) = orig.strip_prefix("/private") {
            out.insert(stripped.to_string());
        }

        if let Some(repo_str) = repo_root.to_str() {
            let mut candidate_roots: Vec<&str> = vec![repo_str];
            if let Some(stripped) = repo_str.strip_prefix("/private") { candidate_roots.push(stripped); }
            for root in candidate_roots {
                if orig.starts_with(root) {
                    let rel = orig[root.len()..].trim_start_matches('/');
                    if !rel.is_empty() {
                        out.insert(rel.to_string());
                        out.insert(format!("./{}", rel));
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    let mut covered_map: HashMap<String, Vec<(u32,u32)>> = HashMap::new();
    let final_path = repo_root.join("coverage/coverage-final.json");
    if !final_path.exists() { tracing::info!("coverage final not found path={}", final_path.display()); return covered_map; }
    let bytes = match std::fs::read(&final_path) { Ok(b) => b, Err(e) => { tracing::warn!("read coverage file failed err={}", e); return covered_map; } };
    let json: serde_json::Value = match serde_json::from_slice(&bytes) { Ok(j) => j, Err(e) => { tracing::warn!("parse coverage json failed err={}", e); return covered_map; } };
    let Some(files_obj) = json.as_object() else { return covered_map; };
    let mut total_files = 0usize; let mut files_with_hits = 0usize; let mut total_hit_lines = 0usize; let mut sample: Vec<String> = Vec::new();
    for (file, data) in files_obj.iter() {
        total_files += 1;
        let (Some(stmt_map), Some(s_map)) = (data.get("statementMap"), data.get("s")) else { continue; };
        let (Some(stmt_obj), Some(s_obj)) = (stmt_map.as_object(), s_map.as_object()) else { continue; };
        let mut line_set: HashSet<u32> = HashSet::new();
        for (id, loc_val) in stmt_obj.iter() {
            if let Some(hit_val) = s_obj.get(id) { if hit_val.as_i64().unwrap_or(0) > 0 { if let Some(loc_obj) = loc_val.as_object() { if let (Some(start_obj), Some(end_obj)) = (loc_obj.get("start"), loc_obj.get("end")) { if let (Some(sl), Some(el)) = (start_obj.get("line").and_then(|v| v.as_u64()), end_obj.get("line").and_then(|v| v.as_u64())) { let (sl_u, el_u) = (sl as u32, el as u32); for l in sl_u..=el_u { line_set.insert(l); } } } } } }
        }
        if line_set.is_empty() { continue; }
        files_with_hits += 1;
        let mut covered_lines: Vec<u32> = line_set.into_iter().collect();
        total_hit_lines += covered_lines.len();
        covered_lines.sort_unstable();
        let mut ranges: Vec<(u32,u32)> = Vec::new(); let mut start = covered_lines[0]; let mut prev = covered_lines[0];
        for &ln in &covered_lines[1..] { if ln == prev + 1 { prev = ln; continue; } ranges.push((start, prev)); start = ln; prev = ln; }
        ranges.push((start, prev));
        if sample.len() < 3 { sample.push(format!("{}:{} lines {} ranges", file, covered_lines.len(), ranges.len())); }
        for variant in normalize_variants(file, repo_root) { covered_map.entry(variant).or_insert_with(|| ranges.clone()); }
    }
    tracing::info!("coverage parsed file={} with_files={} hit_lines={} sample={:?}", total_files, files_with_hits, total_hit_lines, sample);
    if files_with_hits == 0 { tracing::info!("coverage contains zero hit lines (possibly tests not instrumented or mismatch)"); }
    covered_map
}


use crate::types::{UncoveredNode, UncoveredNodeConcise, UncoveredResponse, UncoveredResponseItem};

pub fn parse_node_type(node_type: &str) -> Result<NodeType> {
    let mut chars: Vec<char> = node_type.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
    }
    let titled_case = chars.into_iter().collect::<String>();
    NodeType::from_str(&titled_case)
}

pub fn extract_ref_id(node_data: &NodeData) -> String {
    node_data
        .meta
        .get("ref_id")
        .cloned()
        .unwrap_or_else(|| "placeholder".to_string())
}

pub fn format_node_snippet(
    node_type_str: &str,
    name: &str,
    ref_id: &str,
    weight: usize,
    file: &str,
    start: usize,
    end: usize,
    body: &str,
) -> String {
    format!(
        "<snippet>\nname: {}: {}\nref_id: {}\nweight: {}\nfile: {}\nstart: {}, end: {}\n\n{}\n</snippet>\n\n",
        node_type_str, name, ref_id, weight, file, start, end, body
    )
}

pub fn format_node_concise(node_type_str: &str, name: &str, weight: usize, file: &str) -> String {
    format!(
        "{}: {} (weight: {})\nFile: {}\n\n",
        node_type_str, name, weight, file
    )
}

pub fn create_uncovered_response_items(
    nodes: Vec<(NodeData, usize)>,
    node_type: &NodeType,
    concise: bool,
) -> Vec<UncoveredResponseItem> {
    nodes
        .into_iter()
        .map(|(node_data, weight)| {
            if concise {
                UncoveredResponseItem::Concise(UncoveredNodeConcise {
                    name: node_data.name,
                    file: node_data.file,
                    weight,
                })
            } else {
                let ref_id = extract_ref_id(&node_data);
                UncoveredResponseItem::Full(UncoveredNode {
                    node_type: node_type.to_string(),
                    ref_id,
                    weight,
                    properties: node_data,
                })
            }
        })
        .collect()
}

pub fn format_uncovered_response_as_snippet(response: &UncoveredResponse) -> String {
    let mut text = String::new();

    if let Some(ref functions) = response.functions {
        for item in functions {
            match item {
                UncoveredResponseItem::Full(node) => {
                    text.push_str(&format_node_snippet(
                        &node.node_type,
                        &node.properties.name,
                        &node.ref_id,
                        node.weight,
                        &node.properties.file,
                        node.properties.start,
                        node.properties.end,
                        &node.properties.body,
                    ));
                }
                UncoveredResponseItem::Concise(node) => {
                    text.push_str(&format_node_concise(
                        "Function",
                        &node.name,
                        node.weight,
                        &node.file,
                    ));
                }
            }
        }
    }

    if let Some(ref endpoints) = response.endpoints {
        for item in endpoints {
            match item {
                UncoveredResponseItem::Full(node) => {
                    text.push_str(&format_node_snippet(
                        &node.node_type,
                        &node.properties.name,
                        &node.ref_id,
                        node.weight,
                        &node.properties.file,
                        node.properties.start,
                        node.properties.end,
                        &node.properties.body,
                    ));
                }
                UncoveredResponseItem::Concise(node) => {
                    text.push_str(&format_node_concise(
                        "Endpoint",
                        &node.name,
                        node.weight,
                        &node.file,
                    ));
                }
            }
        }
    }

    text
}
