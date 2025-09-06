use ast::lang::asg::NodeData;
use ast::lang::NodeType;
use shared::Result;
use std::str::FromStr;

use std::collections::{HashMap, HashSet};
use std::path::Path;

pub async fn run_coverage(repo_root: &Path, repo_url: &str) {
    if let Some(root) = repo_root.to_str() {
        let commit = lsp::git::get_commit_hash(root).await.ok().unwrap_or_default();
        if let Err(e) = shared::codecov::run(root, repo_url, &commit) {
            tracing::warn!("coverage run failed repo={} commit={} err={}", repo_url, commit, e);
        }
    }
}

pub fn normalize_coverage_path_variants(original: &str, repo_root: &Path) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();

    let mut push = |s: &str| { if !s.is_empty() { out.insert(s.to_string()); } };

    let mut base_forms = Vec::new();
    base_forms.push(original.to_string());
    if let Some(stripped) = original.strip_prefix("/private") { base_forms.push(stripped.to_string()); }

    // Root related forms
    let (root_str, tail, last_two) = if let Some(r) = repo_root.to_str() {
        let tail = repo_root.file_name().and_then(|s| s.to_str()).map(|s| s.to_string());
        let comps: Vec<&str> = r.split('/').filter(|c| !c.is_empty()).collect();
        let last_two = (comps.len() >= 2).then(|| format!("{}/{}", comps[comps.len()-2], comps[comps.len()-1]));
        (Some(r.to_string()), tail, last_two)
    } else { (None, None, None) };

    let mut root_forms: Vec<String> = Vec::new();
    if let Some(r) = &root_str { root_forms.push(r.clone()); }
    if let Some(r) = &root_str { if let Some(stripped) = r.strip_prefix("/private") { root_forms.push(stripped.to_string()); } }
    if let Some(r) = &root_str { if let Some(stripped) = r.strip_prefix("/tmp/") { root_forms.push(stripped.to_string()); } }
    if let Some(lt) = &last_two { root_forms.push(lt.clone()); }
    root_forms.sort_unstable();
    root_forms.dedup();

    // Always include the full base forms themselves.
    for b in &base_forms { push(b); }

    for base in &base_forms {
        for root in &root_forms {
            if base.starts_with(root) {
                let rel = base[root.len()..].trim_start_matches('/');
                if rel.is_empty() { continue; }
                push(rel);
                push(&format!("./{}", rel));
                if let Some(t) = &tail { push(&format!("{}/{}", t, rel)); push(&format!("./{}/{}", t, rel)); }
                if let Some(lt) = &last_two { push(&format!("{}/{}", lt, rel)); }
                if let Some(r) = &root_str { if let Some(tmp) = r.strip_prefix("/tmp/") { push(&format!("{}/{}", tmp, rel)); } }
            }
        }
    }

    out.into_iter().collect()
}

pub fn covered_line_ranges(repo_root: &Path) -> HashMap<String, Vec<(u32,u32)>> {

    let mut covered: HashMap<String, Vec<(u32,u32)>> = HashMap::new();
    let path = repo_root.join("coverage/coverage-final.json");
    if !path.exists() { return covered; }
    let bytes = match std::fs::read(&path) { Ok(b) => b, Err(_) => return covered };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return covered }; 
    let Some(files_obj) = json.as_object() else { return covered }; 
    let mut files_with_hits = 0usize; let mut total_hit_lines = 0usize; let mut sample: Vec<String> = Vec::new();
    for (file, data) in files_obj.iter() {
        let (Some(stmt_map), Some(s_map)) = (data.get("statementMap"), data.get("s")) else { continue; };
        let (Some(stmt_obj), Some(s_obj)) = (stmt_map.as_object(), s_map.as_object()) else { continue; };
        let mut lines: HashSet<u32> = HashSet::new();
        for (id, loc) in stmt_obj.iter() {
            if s_obj.get(id).and_then(|v| v.as_i64()).unwrap_or(0) <= 0 { continue; }
            if let Some(loc_obj) = loc.as_object() {
                if let (Some(s), Some(e)) = (loc_obj.get("start"), loc_obj.get("end")) {
                    if let (Some(sl), Some(el)) = (s.get("line").and_then(|v| v.as_u64()), e.get("line").and_then(|v| v.as_u64())) {
                        for l in sl as u32..=el as u32 { lines.insert(l); }
                    }
                }
            }
        }
        if lines.is_empty() { continue; }
        files_with_hits += 1;
        let mut covered_lines: Vec<u32> = lines.into_iter().collect();
        total_hit_lines += covered_lines.len();
        covered_lines.sort_unstable();
        let mut ranges: Vec<(u32,u32)> = Vec::new();
        let mut start = covered_lines[0];
        let mut prev = covered_lines[0];
        for &ln in &covered_lines[1..] {
            if ln == prev + 1 { prev = ln; continue; }
            ranges.push((start, prev));
            start = ln; prev = ln;
        }
        ranges.push((start, prev));
        if sample.len() < 3 { sample.push(format!("{}:{} lines {} ranges", file, covered_lines.len(), ranges.len())); }
        for variant in normalize_coverage_path_variants(file, repo_root) { covered.insert(variant, ranges.clone()); }
    }
    tracing::info!("coverage parsed files_with_hits={} hit_lines={} sample={:?}", files_with_hits, total_hit_lines, sample);
    covered
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
