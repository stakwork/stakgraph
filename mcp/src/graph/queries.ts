import { DIMENSIONS } from "../vector/index.js";

export const Data_Bank = "Data_Bank";
export const KEY_INDEX = "data_bank_node_key_index";
export const FULLTEXT_BODY_INDEX = "bodyIndex";
export const FULLTEXT_NAME_INDEX = "nameIndex";
export const VECTOR_INDEX = "vectorIndex";

export const KEY_INDEX_QUERY = `CREATE INDEX ${KEY_INDEX} IF NOT EXISTS FOR (n:${Data_Bank}) ON (n.node_key)`;

const ENGLISH = `OPTIONS {
  indexConfig: {
    \`fulltext.analyzer\`: 'english'
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
${ENGLISH}`;

export const FULLTEXT_NAME_INDEX_QUERY = `CREATE FULLTEXT INDEX ${FULLTEXT_NAME_INDEX}
  IF NOT EXISTS FOR (f:${Data_Bank})
  ON EACH [f.name]
${ENGLISH}`;

export const VECTOR_INDEX_QUERY = `CREATE VECTOR INDEX ${VECTOR_INDEX}
  IF NOT EXISTS FOR (n:${Data_Bank})
  ON n.embeddings
${COSINE}`;

export const DATA_BANK_QUERY = `MATCH (n:${Data_Bank}) RETURN n`;

export const DATA_BANK_BODIES_QUERY = `
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
     END AS startNode,
     direction, labelFilter, depth, trim
WHERE startNode IS NOT NULL

// For bidirectional queries, reduce depth to prevent explosion
WITH startNode, direction, labelFilter, trim,
     CASE WHEN direction = 'both' THEN toInteger(depth/2) ELSE depth END AS effectiveDepth

// Execute separate traversals and combine results
CALL {
    WITH *
    CALL apoc.path.subgraphAll(startNode, {
        relationshipFilter: CASE WHEN direction IN ['down', 'both'] 
                            THEN "RENDERS>|CALLS>|CONTAINS>|HANDLER>|<OPERAND" 
                            ELSE null END,
        minLevel: 1,
        maxLevel: effectiveDepth,
        labelFilter: labelFilter
    })
    YIELD nodes as downNodes, relationships as downRels
    RETURN downNodes, downRels
}

CALL {
    WITH *
    CALL apoc.path.subgraphAll(startNode, {
        relationshipFilter: CASE WHEN direction IN ['up', 'both'] 
                            THEN "<RENDERS|<CALLS|<CONTAINS|<HANDLER|<OPERAND" 
                            ELSE null END,
        minLevel: 1,
        maxLevel: effectiveDepth,
        labelFilter: labelFilter
    })
    YIELD nodes as upNodes, relationships as upRels
    RETURN upNodes, upRels
}

// Combine and deduplicate nodes and relationships
WITH startNode, trim, 
     CASE WHEN direction IN ['down', 'both'] THEN downNodes ELSE [] END + 
     CASE WHEN direction IN ['up', 'both'] THEN upNodes ELSE [] END AS allNodes,
     CASE WHEN direction IN ['down', 'both'] THEN downRels ELSE [] END + 
     CASE WHEN direction IN ['up', 'both'] THEN upRels ELSE [] END AS allRels

// Remove duplicates
WITH startNode, trim,
     apoc.coll.toSet(allNodes) AS uniqueNodes,
     apoc.coll.toSet(allRels) AS uniqueRels

// Filter out trimmed nodes
WITH startNode,
     [n IN uniqueNodes WHERE NOT n.name IN trim] AS filteredNodes,
     uniqueRels, trim

// Filter relationships to only include those between non-trimmed nodes
WITH startNode, filteredNodes,
     [r IN uniqueRels 
      WHERE 
        NOT startNode(r).name IN trim AND
        NOT endNode(r).name IN trim
     ] AS filteredRels,
     trim

// Transform to the format you need
WITH startNode, filteredNodes, 
     [rel IN filteredRels | {
        source: id(startNode(rel)),
        target: id(endNode(rel)),
        type: type(rel),
        properties: properties(rel)
     }] AS relationshipData,
     [node IN filteredNodes WHERE node.file IS NOT NULL | node.file] AS fileNames

// Get imports
MATCH (file:File)-[:CONTAINS]->(import:Import)
WHERE file.file IN fileNames

RETURN startNode,
       filteredNodes AS allNodes,
       relationshipData AS relationships,
       COLLECT(DISTINCT import) AS imports
`;

export const PATH_QUERY = `
WITH $include_tests as include_tests,
     $function_name as function_name,
     $page_name as page_name,
     $depth as depth

// Decide which node to use based on parameters
OPTIONAL MATCH (start_page:Page)
WHERE page_name IS NOT NULL AND start_page.name = page_name

OPTIONAL MATCH (start_func:Function)
WHERE page_name IS NULL AND function_name IS NOT NULL AND start_func.name = function_name

// Combine into a single starting node
WITH include_tests, depth,
     CASE WHEN start_page IS NOT NULL THEN start_page ELSE start_func END as start_node

// Ensure we found a valid starting node
WHERE start_node IS NOT NULL

// Start directly with the chosen node and collect paths
CALL apoc.path.expandConfig(start_node, {
    relationshipFilter: "CALLS>|CONTAINS>|HANDLER>|RENDERS>",
    minLevel: 0,
    maxLevel: depth,
    uniqueness: "NODE_GLOBAL"
}) YIELD path
WITH include_tests, start_node, collect(path) as expanded_paths

// Ensure we always have at least one path with the starting node
WITH include_tests, start_node,
     CASE WHEN size(expanded_paths) > 0
          THEN expanded_paths
          ELSE [apoc.path.create(start_node, [])]
     END as function_paths

// Extract all function nodes from the paths
UNWIND function_paths as path
UNWIND nodes(path) as node
WITH include_tests, start_node, function_paths, collect(DISTINCT node) as all_nodes

// Find additional nodes: classes, traits, and tests
OPTIONAL MATCH (n)-[r]-(related)
WHERE n IN all_nodes
  AND (
    (n:Function AND related:Class) OR
    (n:Function AND related:Trait) OR
    (include_tests AND n:Function AND related:Test) OR
    (include_tests AND n:Function AND related:E2etest) OR
    (n:Function AND related:Page)
  )
WITH include_tests, start_node, function_paths, all_nodes,
     collect(DISTINCT r) as additional_rels

// Create paths for these additional relationships
WITH include_tests, start_node, function_paths, all_nodes,
     [rel IN additional_rels | apoc.path.create(startNode(rel), [rel])] as additional_paths

// Find Files that contain these functions and other nodes
OPTIONAL MATCH (file:File)-[contains:CONTAINS]->(node)
WHERE node IN all_nodes
WITH include_tests, start_node, function_paths, additional_paths, collect(DISTINCT file) as files

// OPTIONAL MATCH for Imports in the same files
OPTIONAL MATCH (file:File)-[imp_rel:CONTAINS]->(import:Import)
WHERE file IN files
WITH include_tests, start_node, function_paths, additional_paths,
     collect(DISTINCT import) as imports

// Create simple paths for imports (if any)
WITH start_node, function_paths, additional_paths, imports
WITH start_node, function_paths, additional_paths,
     [import IN imports | apoc.path.create(import, [])] as import_paths

// Combine all paths
WITH start_node, function_paths + additional_paths + import_paths as all_paths

RETURN
    start_node as function,
    all_paths as paths
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
