use std::collections::{HashMap, HashSet};

use neo4rs::{query as nq, BoltType, Node as NeoNode};
use shared::Result;
use termtree::Tree;
use tiktoken_rs::CoreBPE;

use crate::output::Output;

use super::connection::connect_graph_ops;

const SUBGRAPH_QUERY: &str = r#"
        WITH $node_label AS nodeLabel,
            $node_name as nodeName,
            $node_file as nodeFile,
            $ref_id as refId,
            $direction as direction,
            $label_filter as labelFilter,
            $depth as depth,
            $trim as trim

        CALL {
        WITH refId
        WITH refId WHERE refId IS NOT NULL AND refId <> ''
        MATCH (node {ref_id: refId})
        RETURN node
        UNION
        WITH nodeName, nodeLabel
        WITH nodeName, nodeLabel WHERE nodeName IS NOT NULL AND nodeName <> ''
        MATCH (node {name: nodeName})
        WHERE nodeLabel = '' OR nodeLabel IN labels(node)
        RETURN node
        UNION
        WITH nodeFile, nodeLabel
        WITH nodeFile, nodeLabel WHERE nodeFile IS NOT NULL AND nodeFile <> ''
        MATCH (node)
        WHERE nodeLabel IN labels(node)
            AND node.file IS NOT NULL
            AND node.file CONTAINS nodeFile
        RETURN node
        }

        WITH node AS f, direction, labelFilter, depth, trim
        WHERE f IS NOT NULL

        WITH f, direction, labelFilter, depth, trim,
            CASE WHEN direction IN ["down", "both"] THEN 1 ELSE 0 END AS includeDown

        CALL {
            WITH f, labelFilter, depth, includeDown
            CALL apoc.path.expandConfig(f, {
                relationshipFilter: "RENDERS>|CALLS>|CONTAINS>|HANDLER>|<OPERAND",
                uniqueness: "NODE_PATH",
                minLevel: 1,
                maxLevel: includeDown * depth,
                labelFilter: labelFilter
            })
            YIELD path
            RETURN collect(path) AS downwardPaths
        }

        WITH f, direction, labelFilter, depth, trim, downwardPaths,
            CASE WHEN direction IN ["up", "both"] THEN 1 ELSE 0 END AS includeUp

        CALL {
            WITH f, labelFilter, depth, includeUp
            CALL apoc.path.expandConfig(f, {
                relationshipFilter: "<RENDERS|<CALLS|<CONTAINS|<HANDLER|<OPERAND",
                uniqueness: "NODE_PATH",
                minLevel: 1,
                maxLevel: includeUp * depth,
                labelFilter: labelFilter
            })
            YIELD path
            RETURN collect(path) AS upwardPaths
        }

        WITH f AS startNode,
            downwardPaths + upwardPaths AS paths,
            trim

        WITH startNode,
            CASE
            WHEN size(paths) = 0 THEN []
            ELSE paths
            END AS filteredPaths,
            trim

        WITH startNode,
            filteredPaths,
            trim,
            CASE
            WHEN size(filteredPaths) = 0 THEN [startNode]
            ELSE [startNode] + REDUCE(nodes = [], path IN filteredPaths | nodes + nodes(path))
            END AS pathNodes

        WITH startNode,
            filteredPaths,
            trim,
            [node IN pathNodes WHERE NOT (node.name IN trim) | node] AS filteredNodes

        WITH startNode,
            filteredPaths,
            trim,
            REDUCE(uniqueNodes = [], node IN filteredNodes |
            CASE WHEN node IN uniqueNodes THEN uniqueNodes ELSE uniqueNodes + [node] END
            ) AS allNodes

        WITH startNode,
            allNodes,
            REDUCE(allRels = [], path IN filteredPaths | allRels + relationships(path)) AS pathRels,
            trim

        WITH startNode,
            allNodes,
            REDUCE(uniqueRels = [], rel IN pathRels |
            CASE WHEN rel IN uniqueRels THEN uniqueRels ELSE uniqueRels + [rel] END
            ) AS uniqueRels,
            trim

        WITH startNode,
            allNodes,
            [rel IN uniqueRels | {
            source: id(startNode(rel)),
            target: id(endNode(rel)),
            type: type(rel),
            properties: properties(rel)
            }] AS relationships,
            trim

        RETURN startNode,
            allNodes,
            relationships
"#;

fn node_label(node: &NeoNode, bpe: &CoreBPE) -> (String, u64) {
    let name: String = node.get("name").unwrap_or_default();
    let label = if name.is_empty() {
        node.get::<String>("file").unwrap_or_else(|_| "?".to_string())
    } else {
        name
    };
    let token_count: Option<i64> = node.get("token_count").ok();
    let (count, from_stored) = match token_count {
        Some(t) if t > 0 => (t as u64, true),
        _ => {
            let body: String = node.get("body").unwrap_or_default();
            if body.is_empty() {
                (0, false)
            } else {
                (bpe.encode_with_special_tokens(&body).len() as u64, false)
            }
        }
    };
    let _ = from_stored; // both paths now produce exact counts
    if count > 0 {
        (format!("{} ({})", label, count), count)
    } else {
        (label, 0)
    }
}

pub(super) async fn run_map(
    name: &str,
    node_type: Option<&str>,
    direction: &str,
    depth: usize,
    tests: bool,
    trim: &[String],
    out: &mut Output,
) -> Result<()> {
    let ops = connect_graph_ops().await?;
    let connection = ops.graph.ensure_connected().await?;

    let node_label_param = node_type.unwrap_or("").to_string();

    // Build label filter: exclude infrastructure nodes, optionally exclude test nodes
    let mut excluded = vec![
        "-File", "-Directory", "-Repository", "-Library", "-Import", "-Language", "-Package",
    ];
    if !tests {
        excluded.push("-UnitTest");
        excluded.push("-IntegrationTest");
        excluded.push("-E2eTest");
    }
    let label_filter = excluded.join("|");

    let trim_bolt: Vec<BoltType> = trim
        .iter()
        .map(|s| BoltType::String(s.clone().into()))
        .collect();

    let q = nq(SUBGRAPH_QUERY)
        .param("node_label", node_label_param.clone())
        .param("node_name", name)
        .param("node_file", "")
        .param("ref_id", "")
        .param("direction", direction)
        .param("label_filter", label_filter)
        .param("depth", depth as i64)
        .param("trim", trim_bolt.as_slice());

    let mut result = connection.execute(q).await?;
    let row = match result.next().await? {
        Some(r) => r,
        None => {
            out.writeln(format!("No node found with name {:?}", name))?;
            return Ok(());
        }
    };

    let start_node: NeoNode = row.get("startNode").map_err(|e| {
        shared::Error::internal(format!("missing startNode: {}", e))
    })?;
    let all_nodes: Vec<NeoNode> = row.get("allNodes").map_err(|e| {
        shared::Error::internal(format!("missing allNodes: {}", e))
    })?;

    #[derive(Debug)]
    struct Rel {
        source: i64,
        target: i64,
        rel_type: String,
    }

    let relationships: Vec<Rel> = {
        use neo4rs::BoltMap;
        let raw: Vec<BoltMap> = row.get("relationships").unwrap_or_default();
        raw.into_iter()
            .filter_map(|m| {
                let source: i64 = m.get::<i64>("source").ok()?;
                let target: i64 = m.get::<i64>("target").ok()?;
                let rel_type: String = m.get::<String>("type").ok()?;
                Some(Rel { source, target, rel_type })
            })
            .collect()
    };

    let start_id = start_node.id();

    let bpe = tiktoken_rs::cl100k_base()
        .map_err(|e| shared::Error::internal(format!("tiktoken init failed: {}", e)))?;

    // Build id → label map, accumulating token counts from every node
    let mut id_to_label: HashMap<i64, String> = HashMap::new();
    let mut total_tokens: u64 = 0;

    let (start_label, start_tokens) = node_label(&start_node, &bpe);
    id_to_label.insert(start_id, start_label);
    total_tokens += start_tokens;

    for node in &all_nodes {
        let nid = node.id();
        if nid == start_id { continue; }
        let (label, tokens) = node_label(node, &bpe);
        id_to_label.insert(nid, label);
        total_tokens += tokens;
    }

    // Build parent → children adjacency, respecting OPERAND reversal
    let reverse_rels: HashSet<&str> = ["OPERAND"].iter().copied().collect();
    let mut children: HashMap<i64, Vec<i64>> = HashMap::new();

    for rel in &relationships {
        let (parent, child) = if direction == "up" {
            if reverse_rels.contains(rel.rel_type.as_str()) {
                (rel.source, rel.target)
            } else {
                (rel.target, rel.source)
            }
        } else {
            // "down" or "both"
            if reverse_rels.contains(rel.rel_type.as_str()) {
                (rel.target, rel.source)
            } else {
                (rel.source, rel.target)
            }
        };
        children.entry(parent).or_default().push(child);
    }

    // Build termtree recursively (BFS to avoid cycles)
    fn build_subtree(
        id: i64,
        id_to_label: &HashMap<i64, String>,
        children: &HashMap<i64, Vec<i64>>,
        visited: &mut HashSet<i64>,
    ) -> Tree<String> {
        let label = id_to_label.get(&id).cloned().unwrap_or_else(|| id.to_string());
        let mut tree = Tree::new(label);
        if visited.contains(&id) {
            return tree;
        }
        visited.insert(id);
        if let Some(kids) = children.get(&id) {
            for &kid in kids {
                if kid != id {
                    tree.push(build_subtree(kid, id_to_label, children, visited));
                }
            }
        }
        tree
    }

    let mut visited = HashSet::new();
    let tree = build_subtree(start_id, &id_to_label, &children, &mut visited);

    // Orphaned nodes (not reachable from root via relationships) go as direct children of root
    let mut root_tree = tree;
    let placed: HashSet<i64> = visited.clone();
    for (&id, label) in &id_to_label {
        if id != start_id && !placed.contains(&id) {
            root_tree.push(Tree::new(label.clone()));
        }
    }

    out.writeln(root_tree.to_string())?;
    out.writeln(format!("total tokens: {}", total_tokens))?;
    Ok(())
}
