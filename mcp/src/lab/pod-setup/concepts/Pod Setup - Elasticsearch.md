---
description: "Elasticsearch beside the app: a single-node dev container, reached at 127.0.0.1."
parent: Pod Setup - Local Services
---
Elasticsearch beside the app: a single-node dev container, reached at 127.0.0.1.

- An `elasticsearch` compose service (`image:
  docker.elastic.co/elasticsearch/elasticsearch:8.19.23`, `9.5.5` when the
  repository's client is 9.x, or the version the repository names),
  `ports: ["9200:9200"]`, with `discovery.type:
  single-node`, `xpack.security.enabled: "false"` and a modest heap
  (`ES_JAVA_OPTS: -Xms512m -Xmx512m`).
- The app's Elasticsearch URL is `http://127.0.0.1:9200`.
- OpenSearch the same way, with its own image
  (`opensearchproject/opensearch:2`, or the version the repository
  names), `OPENSEARCH_JAVA_OPTS` for the heap, and
  `DISABLE_SECURITY_PLUGIN: "true"` and `DISABLE_INSTALL_DEMO_CONFIG:
  "true"` in place of the `xpack` setting, which OpenSearch does not know
  and will not start with.
