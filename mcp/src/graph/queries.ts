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
RETURN f
`;

export const REF_IDS_LIST_QUERY = `
WITH $ref_ids AS refIdList
MATCH (n)
WHERE n.ref_id IN refIdList
RETURN n
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
WITH node, gds.similarity.cosine(node.embeddings, $embeddings) AS score
WHERE score >= $similarityThreshold
RETURN node, score
ORDER BY score DESC
LIMIT toInteger($limit)
`;

export const SUBGRAPH_QUERY = `
WITH $node_label AS nodeLabel,
     $node_name as nodeName,
     $ref_id as refId,
     $direction as direction,
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
     direction, labelFilter, depth, trim
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
        maxLevel: includeDown * depth, // Will be 0 if not including down paths
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
        maxLevel: includeUp * depth, // Will be 0 if not including up paths
        labelFilter: labelFilter
    })
    YIELD path
    RETURN collect(path) AS upwardPaths
}

// Combine the paths
WITH f AS startNode, 
     downwardPaths + upwardPaths AS paths,
     trim

// Rest of query remains the same
UNWIND paths AS path
WITH startNode, path, trim
WHERE NONE(n IN nodes(path) WHERE n.name IN trim)

WITH startNode,
     COLLECT(DISTINCT path) AS filteredPaths,
     trim

UNWIND filteredPaths AS path
UNWIND nodes(path) AS node
WITH startNode, filteredPaths, trim, COLLECT(DISTINCT node) AS allNodes

UNWIND filteredPaths AS path
UNWIND relationships(path) AS rel
WITH startNode, allNodes, COLLECT(DISTINCT {
    source: id(startNode(rel)),
    target: id(endNode(rel)),
    type: type(rel),
    properties: properties(rel)
}) AS relationships, trim

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
