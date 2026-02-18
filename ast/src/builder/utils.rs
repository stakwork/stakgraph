use crate::lang::{asg::NodeData, graphs::NodeType};
use crate::lang::{Graph, Node};
use crate::repo::Repo;
use crate::utils::create_node_key;
use lsp::{strip_tmp, Language};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const MAX_FILE_SIZE: u64 = 500_000;

#[cfg(feature = "openssl")]
pub fn filter_by_revs<G: Graph>(root: &str, revs: Vec<String>, graph: G, lang_kind: Language) -> G {
    if revs.is_empty() {
        return graph;
    }
    match crate::repo::check_revs_files(root, revs) {
        Some(final_filter) => graph.create_filtered_graph(&final_filter, lang_kind),
        None => graph,
    }
}

#[cfg(not(feature = "openssl"))]
pub fn filter_by_revs<G: Graph>(
    _root: &str,
    _revs: Vec<String>,
    graph: G,
    _lang_kind: Language,
) -> G {
    graph
}

pub fn _filenamey(f: &Path) -> String {
    let full = f.display().to_string();
    if !f.starts_with("/tmp/") {
        return full;
    }
    let mut parts = full.split("/").collect::<Vec<&str>>();
    parts.drain(0..4);
    parts.join("/")
}

pub fn get_page_name(path: &str) -> Option<String> {
    let parts = path.split("/").collect::<Vec<&str>>();
    parts.last()?;
    Some(parts.last().unwrap().to_string())
}

pub fn combine_import_sections(nodes: Vec<NodeData>) -> Vec<NodeData> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let import_name = create_node_key(&Node::new(NodeType::Import, nodes[0].clone()));

    let mut seen_starts = HashSet::new();
    let mut unique_nodes = Vec::new();
    for node in nodes {
        if !seen_starts.contains(&node.start) {
            seen_starts.insert(node.start);
            unique_nodes.push(node);
        }
    }

    // Use the file from the first node
    let file = if !unique_nodes.is_empty() {
        unique_nodes[0].file.clone()
    } else {
        String::new()
    };

    vec![NodeData {
        name: import_name,
        file,
        start: unique_nodes[0].start,
        end: unique_nodes.last().unwrap().end,
        ..Default::default()
    }]
}
pub fn is_allowed_file(path: &Path, lang: &Language) -> bool {
    let fname = path.display().to_string();
    if lang
        .pkg_files()
        .iter()
        .any(|pkg_file| fname.ends_with(pkg_file))
    {
        return true;
    }
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        if lang.exts().contains(&ext) {
            return true;
        }
    }
    false
}

impl Repo {
    pub fn prepare_file_data(&self, path: &str, code: &str) -> NodeData {
        let mut file_data = NodeData::in_file(path);
        let filename = path.split('/').next_back().unwrap_or(path);
        file_data.name = filename.to_string();
        file_data.end = code.lines().count().saturating_sub(1);
        file_data.hash = Some(sha256::digest(code));
        file_data
    }
    pub fn get_parent_info(&self, path: &Path) -> (NodeType, String) {
        let stripped_path = strip_tmp(path).display().to_string();

        let root_no_tmp = strip_tmp(&self.root).display().to_string();
        let mut dir_no_root = stripped_path
            .strip_prefix(&root_no_tmp)
            .unwrap_or(&stripped_path);
        dir_no_root = dir_no_root.trim_start_matches('/');

        let filepath = path.display().to_string();
        if dir_no_root.contains("/") {
            let mut paths: Vec<&str> = filepath.split('/').collect();
            paths.pop();
            let dirpath = paths.join("/");
            let fin = strip_tmp(&PathBuf::from(dirpath)).display().to_string();
            (NodeType::Directory, fin)
        } else {
            let repo_file = strip_tmp(&self.root).display().to_string();
            (NodeType::Repository, repo_file)
        }
    }
}
