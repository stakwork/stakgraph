### fulltext

create:

```
CREATE FULLTEXT INDEX bodyIndex FOR (f:Repository|Directory|File|Import|Class|Library|Function|Test|E2eTest|Endpoint|Request|Datamodel|Page)
ON EACH [f.body]
OPTIONS {
  indexConfig: {
    `fulltext.analyzer`: 'english'
  }
}
```

query:

CALL db.index.fulltext.queryNodes("nameBodyFileIndex", 'body:"margin-bottom: 5px"') YIELD node, score

```
curl "http://localhost:3000/search?query=margin-bottom:%205px"

curl "http://localhost:3000/search?query=NewRouter"
```

composite:

```
CALL db.index.fulltext.queryNodes("bodyIndex", "leaderboard") YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT 25
```

drop:

```
DROP INDEX bodyIndex
```

by node type:

```
WITH ["Function"] as node_types
CALL db.index.fulltext.queryNodes('bodyIndex', 'leaderboard') YIELD node, score
WITH node, score
WHERE
  (NOT exists(node_types) OR size(node_types) = 0)
  OR
  ANY(label IN labels(node) WHERE label IN node_types)
RETURN node, score
ORDER BY score DESC
LIMIT toInteger($limit)
```
