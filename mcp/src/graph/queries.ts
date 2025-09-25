import { DIMENSIONS } from "../vector/index.js";

export const Data_Bank = "Data_Bank";
export const KEY_INDEX = "data_bank_node_key_index";
export const FULLTEXT_BODY_INDEX = "bodyIndex";
export const FULLTEXT_NAME_INDEX = "nameIndex";
export const FULLTEXT_COMPOSITE_INDEX = "nameBodyFileIndex";
export const VECTOR_INDEX = "vectorIndex";

export const KEY_INDEX_QUERY = `CREATE INDEX ${KEY_INDEX} IF NOT EXISTS FOR (n:${Data_Bank}) ON (n.node_key)`;

const ENGLISH_ANALYZER = `OPTIONS {
  indexConfig: {
    \`fulltext.analyzer\`: 'english'
  }
}`;

// less aggressive analyzer
const STANDARD_ANALYZER = `OPTIONS {
  indexConfig: {
    \`fulltext.analyzer\`: 'standard'
  }
}`;

const COSINE = `OPTIONS {
  indexConfig: {
    \`vector.dimensions\`: ${DIMENSIONS},
    \`vector.similarity_function\`: 'cosine'
  }
}`;

export const FULLTEXT_BODY_INDEX_QUERY = `CREATE FULLTEXT INDEX ${FULLTEXT_BODY_INDEX}
  IF NOT EXISTS FOR (f:${Data_Bank})
  ON EACH [f.body]
${STANDARD_ANALYZER}`;

export const FULLTEXT_NAME_INDEX_QUERY = `CREATE FULLTEXT INDEX ${FULLTEXT_NAME_INDEX}
  IF NOT EXISTS FOR (f:${Data_Bank})
  ON EACH [f.name]
${STANDARD_ANALYZER}`;

export const FULLTEXT_COMPOSITE_INDEX_QUERY = `
CREATE FULLTEXT INDEX ${FULLTEXT_COMPOSITE_INDEX}
  IF NOT EXISTS FOR (f:${Data_Bank})
  ON EACH [f.name, f.body, f.file]
${STANDARD_ANALYZER}`;

export const VECTOR_INDEX_QUERY = `CREATE VECTOR INDEX ${VECTOR_INDEX}
  IF NOT EXISTS FOR (n:${Data_Bank})
  ON n.embeddings
${COSINE}`;

export const DATA_BANK_QUERY = `MATCH (n:${Data_Bank}) RETURN n`;

export const DATA_BANK_BODIES_QUERY_NO_EMBEDDINGS = `
  MATCH (n:${Data_Bank})
  WHERE n.embeddings IS NULL
    AND (($do_files = true) OR NOT n:File)
  RETURN n.node_key as node_key, n.body as body
  SKIP toInteger($skip) LIMIT toInteger($limit)
`;

export const UPDATE_EMBEDDINGS_QUERY = `
MATCH (n:${Data_Bank} {node_key: $node_key})
SET n.embeddings = $embeddings
`;

export const BULK_UPDATE_EMBEDDINGS_QUERY = `
UNWIND $batch as item
MATCH (n:${Data_Bank} {node_key: item.node_key})
SET n.embeddings = item.embeddings
`;

export const DATA_BANK_BODIES_QUERY_NO_TOKEN_COUNT = `
  MATCH (n:${Data_Bank})
  WHERE n.token_count IS NULL
    AND (($do_files = true) OR NOT n:File)
  RETURN n.node_key as node_key, n.body as body
`;

export const UPDATE_TOKEN_COUNT_QUERY = `
MATCH (n:${Data_Bank} {node_key: $node_key})
SET n.token_count = $token_count
`;

export const CREATE_HINT_QUERY = `
MERGE (n:Hint:${Data_Bank} {node_key: $node_key})
ON CREATE SET n.ref_id = randomUUID(), n.date_added_to_graph = $ts, n.persona = $persona
SET n.name = $name, n.file = $file, n.body = $body, n.start = 0, n.end = 0, n.question = $question, n.embeddings = $embeddings
RETURN n
`;

export const GET_HINT_QUERY = `
MATCH (n:Hint {node_key: $node_key}) RETURN n
`;

export const CREATE_PROMPT_QUERY = `
MERGE (n:Prompt:${Data_Bank} {node_key: $node_key})
ON CREATE SET n.ref_id = randomUUID(), n.date_added_to_graph = $ts
SET n.name = $name, n.file = $file, n.body = $body, n.start = 0, n.end = 0, n.question = $question, n.embeddings = $embeddings
RETURN n
`;

export const GET_PROMPT_QUERY = `
MATCH (n:Prompt {node_key: $node_key}) RETURN n
`;

export const GET_CONNECTED_HINTS_QUERY = `
MATCH (p:Prompt {ref_id: $prompt_ref_id})-[r]->(h:Hint)
RETURN h
`;

export const HINTS_WITHOUT_SIBLINGS_QUERY = `
MATCH (h:Hint)
WHERE NOT (h)-[:SIBLING]-(:Hint)
RETURN h
`;

export const CREATE_SIBLING_EDGE_QUERY = `
MATCH (a:Hint {ref_id: $source_ref_id})
MATCH (b:Hint {ref_id: $target_ref_id})
WHERE a.ref_id <> b.ref_id
MERGE (a)-[r:SIBLING]->(b)
RETURN r
`;

export const GET_HINT_SIBLINGS_QUERY = `
MATCH (h:Hint {ref_id: $ref_id})-[:SIBLING]-(s:Hint)
RETURN s
`;

export const SET_HINT_PERSONA_QUERY = `
MATCH (h:Hint {ref_id: $ref_id})
SET h.persona = $persona
`;

export const FIND_NODES_BY_NAME_QUERY = `
MATCH (n:{LABEL})
WHERE n.name = $name
RETURN n
LIMIT 5
`;
export const FIND_FILE_NODES_BY_PATH_QUERY = `
MATCH (n:File)
WHERE n.file ENDS WITH $file_path
RETURN n
LIMIT 5
`;

export const CREATE_HINT_EDGES_BY_REF_IDS_QUERY = `
MATCH (h)
WHERE h.ref_id = $hint_ref_id AND (h:Hint OR h:Prompt)
UNWIND $weighted_ref_ids AS weighted_ref
MATCH (t {ref_id: weighted_ref.ref_id})
MERGE (h)-[r:USES]->(t)
SET r.relevancy = weighted_ref.relevancy
RETURN collect(distinct t.ref_id) as refs
`;

export const PKGS_QUERY = `
MATCH (file:File)
WHERE file.name ENDS WITH 'Cargo.toml'
   OR file.name ENDS WITH 'go.mod'
   OR file.name ENDS WITH 'package.json'
   OR file.name ENDS WITH 'requirements.txt'
   OR file.name ENDS WITH 'Gemfile'
RETURN DISTINCT file
`;

export const LIST_QUERY = `
WITH $node_label AS nodeLabel
MATCH (f)
WHERE any(label IN labels(f) WHERE label = nodeLabel)
AND
CASE
  WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
  ELSE f.file IS NOT NULL AND ANY(ext IN $extensions WHERE f.file ENDS WITH ext)
END
RETURN f
`;

export const REF_IDS_LIST_QUERY = `
WITH $ref_ids AS refIdList
MATCH (n)
WHERE n.ref_id IN refIdList
AND
CASE
  WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
  ELSE n.file IS NOT NULL AND ANY(ext IN $extensions WHERE n.file ENDS WITH ext)
END
RETURN n
`;

export const MULTI_TYPE_LATEST_PER_TYPE_QUERY = `
UNWIND $labels AS lbl
MATCH (n)
WHERE any(l IN labels(n) WHERE l = lbl)
  AND ($since IS NULL OR (n.date_added_to_graph IS NOT NULL AND toFloat(n.date_added_to_graph) >= $since))
  AND (
    $extensions IS NULL OR size($extensions) = 0 OR
    (n.file IS NOT NULL AND ANY(ext IN $extensions WHERE n.file ENDS WITH ext))
  )
WITH lbl, n, coalesce(toFloat(n.date_added_to_graph), 0) AS ts
ORDER BY lbl, ts DESC, n.node_key
WITH lbl, collect(n)[0..$limit_per_type] AS nodes
UNWIND nodes AS n
RETURN DISTINCT n;
`;

export const MULTI_TYPE_LATEST_TOTAL_QUERY = `
MATCH (n)
WHERE ($labelsSize = 0 OR any(l IN labels(n) WHERE l IN $labels))
  AND ($since IS NULL OR (n.date_added_to_graph IS NOT NULL AND toFloat(n.date_added_to_graph) >= $since))
  AND (
    $extensions IS NULL OR size($extensions) = 0 OR
    (n.file IS NOT NULL AND ANY(ext IN $extensions WHERE n.file ENDS WITH ext))
  )
WITH n, coalesce(toFloat(n.date_added_to_graph),0) AS ts
ORDER BY ts DESC, n.node_key
LIMIT $limit_total
RETURN n;
`;

export const EDGES_BETWEEN_NODE_KEYS_QUERY = `
MATCH (a)-[r]->(b)
WHERE a.node_key IN $keys AND b.node_key IN $keys
RETURN r, a, b;
`;

export const FILES_QUERY = `
MATCH path = (d:Directory)-[:CONTAINS*0..]->(node)
WHERE (node:Directory OR node:File)
AND (
  $prefix IS NULL
  OR $prefix = ''
  OR d.file STARTS WITH $prefix
)
RETURN path
LIMIT toInteger($limit)
`;

export const FILE_QUERY = `
MATCH (n:File) WHERE n.name ENDS WITH $file_name return n
`;

export const REPOSITORIES_QUERY = `
MATCH (r:Repository) RETURN r
`;

export const FILE_ENDS_WITH_QUERY = `
MATCH (f:File)
WHERE f.file ENDS WITH $file_name
RETURN f
`;

export const PAGES_QUERY = `
MATCH (page:Page)
RETURN DISTINCT page
`;

export const COMPONENTS_QUERY = `
MATCH (f:Function)
WHERE
  // Check if first character is uppercase (ASCII A-Z range)
  f.name =~ '^[A-Z].*'
  // Check if file ends with tsx or jsx
  AND (f.file ENDS WITH '.tsx' OR f.file ENDS WITH '.jsx')
RETURN f as component
`;

export const SEARCH_QUERY_SIMPLE = `
CALL db.index.fulltext.queryNodes('${FULLTEXT_BODY_INDEX}', $query) YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT toInteger($limit)
`;

const NODE_TYPES = `WITH node, score
WHERE
  CASE
    WHEN $node_types IS NULL OR size($node_types) = 0 THEN true
    ELSE ANY(label IN labels(node) WHERE label IN $node_types)
  END
  AND
  CASE
    WHEN $skip_node_types IS NULL OR size($skip_node_types) = 0 THEN true
    ELSE NOT ANY(label IN labels(node) WHERE label IN $skip_node_types)
  END
  AND
  CASE
    WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
    ELSE node.file IS NOT NULL AND ANY(ext IN $extensions WHERE node.file ENDS WITH ext)
  END
RETURN node, score
ORDER BY score DESC
LIMIT toInteger($limit)`;

export const SEARCH_QUERY_BODY = `
CALL db.index.fulltext.queryNodes('${FULLTEXT_BODY_INDEX}', $query) YIELD node, score
${NODE_TYPES}
`;

export const SEARCH_QUERY_NAME = `
CALL db.index.fulltext.queryNodes('${FULLTEXT_NAME_INDEX}', $query) YIELD node, score
${NODE_TYPES}
`;

export const SEARCH_QUERY_COMPOSITE = `
CALL db.index.fulltext.queryNodes('${FULLTEXT_COMPOSITE_INDEX}', $query) YIELD node, score
${NODE_TYPES}
`;

export const VECTOR_SEARCH_QUERY = `
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
`;

export const SUBGRAPH_QUERY = `
WITH $node_label AS nodeLabel,
     $node_name as nodeName,
     $node_file as nodeFile,
     $ref_id as refId,
     $direction as direction,
     $label_filter as labelFilter,
     $depth as depth,
     $trim as trim

// Find by ref_id (highest precedence)
CALL {
  WITH refId
  WITH refId WHERE refId IS NOT NULL AND refId <> ''
  MATCH (node {ref_id: refId})
  RETURN node
  UNION
  // Find by name + label
  WITH nodeName, nodeLabel
  WITH nodeName, nodeLabel WHERE nodeName IS NOT NULL AND nodeName <> ''
  MATCH (node {name: nodeName})
  WHERE nodeLabel IN labels(node)
  RETURN node
  UNION
  // Find by file + label  
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

// First handle "down" direction
WITH f, direction, labelFilter, depth, trim,
     CASE WHEN direction IN ["down", "both"] THEN 1 ELSE 0 END AS includeDown
     
// Get downward paths conditionally
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

// Now handle "up" direction
WITH f, direction, labelFilter, depth, trim, downwardPaths,
     CASE WHEN direction IN ["up", "both"] THEN 1 ELSE 0 END AS includeUp
     
// Get upward paths conditionally
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

// Combine the paths and ALWAYS include the start node
WITH f AS startNode, 
     downwardPaths + upwardPaths AS paths,
     trim

// Handle the case where no paths exist (isolated node)
WITH startNode, 
     CASE 
       WHEN size(paths) = 0 THEN []
       ELSE paths
     END AS filteredPaths,
     trim

// Always include the start node in allNodes
WITH startNode,
     filteredPaths,
     trim,
     // Extract all nodes from paths, but always include startNode
     CASE 
       WHEN size(filteredPaths) = 0 THEN [startNode]
       ELSE [startNode] + REDUCE(nodes = [], path IN filteredPaths | nodes + nodes(path))
     END AS pathNodes

// Remove duplicates and apply trim filter
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

// Extract relationships (will be empty if no paths)
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

WITH startNode, allNodes, relationships,
     [node IN allNodes WHERE node.file IS NOT NULL | node.file] AS fileNames

OPTIONAL MATCH (file:File)-[:CONTAINS]->(import:Import)
WHERE file.file IN fileNames

RETURN startNode,
       allNodes,
       relationships,
       COLLECT(DISTINCT import) AS imports,
       COLLECT(DISTINCT file) AS files
`;

export const REPO_SUBGRAPH_QUERY = `
WITH $node_label AS nodeLabel,
     $node_name as nodeName,
     $ref_id as refId,
     $label_filter as labelFilter,
     $depth as depth,
     $trim as trim

// Find the start node using either ref_id or name+label
OPTIONAL MATCH (fByName {name: nodeName})
WHERE any(label IN labels(fByName) WHERE label = nodeLabel)

OPTIONAL MATCH (fByRefId {ref_id: refId})
WHERE refId <> ''

// ref_id takes precedence over name+label
WITH CASE
       WHEN fByRefId IS NOT NULL THEN fByRefId
       ELSE fByName
     END AS f,
     labelFilter, depth, trim
WHERE f IS NOT NULL

// Get downward paths using CONTAINS relationship only
CALL apoc.path.expandConfig(f, {
    relationshipFilter: "CONTAINS>",
    uniqueness: "NODE_PATH",
    minLevel: 1,
    maxLevel: depth,
    labelFilter: labelFilter
})
YIELD path

// Filter out paths containing trimmed nodes
WITH f AS startNode, path, trim
WHERE NONE(n IN nodes(path) WHERE n.name IN trim)

WITH startNode, COLLECT(DISTINCT path) AS filteredPaths

// Extract all nodes from paths
UNWIND filteredPaths AS path
UNWIND nodes(path) AS node
WITH startNode, filteredPaths, COLLECT(DISTINCT node) AS allNodes

// Extract all relationships from paths
UNWIND filteredPaths AS path
UNWIND relationships(path) AS rel
WITH startNode, allNodes, COLLECT(DISTINCT {
    source: id(startNode(rel)),
    target: id(endNode(rel)),
    type: type(rel),
    properties: properties(rel)
}) AS relationships

RETURN startNode,
       allNodes,
       relationships
`;

export const SHORTEST_PATH_QUERY = `
MATCH (start {node_key: $start_node_key}),
      (end {node_key: $end_node_key})
MATCH path = shortestPath((start)-[*]-(end))
WHERE ALL(node IN nodes(path) WHERE
    node:Page OR
    node:Function OR
    node:Request OR
    node:Endpoint OR
    node:Datamodel)
RETURN path
`;

// export const SHORTEST_PATH_QUERY = `
// MATCH (start {node_key: $start_node_key}), (end {node_key: $end_node_key})
// MATCH path = shortestPath((start)-[*]-(end))
// RETURN path
// `;

export const SHORTEST_PATH_QUERY_REF_ID = `
MATCH (start {ref_id: $start_ref_id}), (end {ref_id: $end_ref_id})
MATCH path = shortestPath((start)-[*]-(end))
WHERE ALL(node IN nodes(path) WHERE
    node:Page OR
    node:Function OR
    node:Request OR
    node:Endpoint OR
    node:Datamodel)
RETURN path
`;

export const RULES_FILES_QUERY = `
MATCH (f:File)
WHERE
  f.name ENDS WITH '.windsurfrules' OR
  f.name ENDS WITH '.cursorrules' OR
  f.name ENDS WITH 'CLAUDE.md' OR
  f.file CONTAINS '/.cursor/rules/' OR
  f.name ENDS WITH 'AGENTS.md' OR
  f.name ENDS WITH '.goosehints' OR
  f.file ENDS WITH '.windsurfrules' OR
  f.file ENDS WITH '.cursorrules' OR
  f.file ENDS WITH 'CLAUDE.md' OR
  f.file ENDS WITH 'AGENTS.md' OR
  f.file ENDS WITH '.goosehints'
RETURN f
ORDER BY f.file
`;

// TODO: Add support for other languages (e.g., Python's os.environ.get)
export const ENV_VARS_QUERY = `
MATCH (n)
WHERE (n:Function OR n:Var)
  AND (
    n.body CONTAINS 'process.env' OR
    n.body CONTAINS 'os.Getenv' OR
    n.body CONTAINS 'os.LookupEnv' OR
    n.body CONTAINS 'ENV[' OR
    n.body CONTAINS 'std::env::var'
  )
RETURN n
`;

export const EDGES_BY_TYPE_QUERY = `
MATCH (source)-[r]->(target)
WHERE
  CASE
    WHEN $edge_types IS NULL OR size($edge_types) = 0 THEN true
    ELSE type(r) IN $edge_types
  END
  AND source.ref_id IS NOT NULL
  AND target.ref_id IS NOT NULL
  AND
  CASE
    WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
    ELSE (source.file IS NOT NULL AND ANY(ext IN $extensions WHERE source.file ENDS WITH ext))
         OR (target.file IS NOT NULL AND ANY(ext IN $extensions WHERE target.file ENDS WITH ext))
  END
RETURN r, source.ref_id AS source_ref_id, target.ref_id AS target_ref_id, type(r) AS edge_type, properties(r) AS properties
ORDER BY edge_type, source_ref_id
LIMIT toInteger($limit)
`;

export const EDGES_BY_REF_IDS_QUERY = `
UNWIND $ref_ids AS refId
MATCH (source)-[r]->(target)
WHERE (source.ref_id = refId OR target.ref_id = refId)
  AND source.ref_id IS NOT NULL
  AND target.ref_id IS NOT NULL
  AND
  CASE
    WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
    ELSE (source.file IS NOT NULL AND ANY(ext IN $extensions WHERE source.file ENDS WITH ext))
         OR (target.file IS NOT NULL AND ANY(ext IN $extensions WHERE target.file ENDS WITH ext))
  END
RETURN r, source.ref_id AS source_ref_id, target.ref_id AS target_ref_id, type(r) AS edge_type, properties(r) AS properties
ORDER BY edge_type, source_ref_id
LIMIT toInteger($limit)
`;

export const ALL_EDGES_QUERY = `
MATCH (source)-[r]->(target)
WHERE source.ref_id IS NOT NULL
  AND target.ref_id IS NOT NULL
  AND
  CASE
    WHEN $extensions IS NULL OR size($extensions) = 0 THEN true
    ELSE (source.file IS NOT NULL AND ANY(ext IN $extensions WHERE source.file ENDS WITH ext))
         OR (target.file IS NOT NULL AND ANY(ext IN $extensions WHERE target.file ENDS WITH ext))
  END
RETURN r, source.ref_id AS source_ref_id, target.ref_id AS target_ref_id, type(r) AS edge_type, properties(r) AS properties
ORDER BY edge_type, source_ref_id
LIMIT toInteger($limit)
`;
/*

CALL db.index.fulltext.queryNodes('nameIndex', 'bounty') YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT 25

*/

/*
MATCH (start {node_key: 'p-stakworksphinxtribesfrontendsrcpagesindextsx'}), (end {node_key: 'person-stakworksphinxtribesdbstructsgo'})
CALL apoc.algo.shortestPath(start, end, '')
YIELD path
RETURN path

MATCH (start {node_key: 'p-stakworksphinxtribesfrontendsrcpagesindextsx'}), (end {node_key: 'person-stakworksphinxtribesdbstructsgo'})
MATCH path = shortestPath((start)-[*]-(end))
RETURN path

MATCH (start {ref_id: 'bb6bab51-018b-41ad-948e-d7bb53179e57'}), (end {ref_id: '426dd007-e2e9-475a-b3a7-36928042bf7b'})
MATCH path = shortestPath((start)-[*]-(end))
RETURN path

*/
