---
description: "Elasticsearch beside the app: a single-node dev container, reached at localhost."
parent: Pod Setup - Local Services
---
Elasticsearch beside the app: a single-node dev container, reached at localhost.

- An `elasticsearch` compose service (`image:
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0`, or the version the
  repository names), `ports: ["9200:9200"]`, with `discovery.type:
  single-node`, `xpack.security.enabled: "false"` and a modest heap
  (`ES_JAVA_OPTS: -Xms512m -Xmx512m`).
- The app's Elasticsearch URL is `http://localhost:9200`.
- OpenSearch the same way, with its own image.
