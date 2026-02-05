# MCP Ingestion Agents

## Mocks Agent
- Endpoint: GET /mocks
- Purpose: Scans a repo for 3rd‑party service integrations and records mock coverage in the graph.
- Params:
  - repo_url (query, required)
  - username (query, optional)
  - pat (query, optional)
  - sync (query, optional; "true"/"1" for incremental sync)
- Responses:
  - 200: { request_id, status: "pending" }
  - 400: { error: "Missing repo_url" }
  - 500: { error: "Internal server error" }
- Standalone call path:
  - Triggered when ProcessBody.mocks passes should_call_mcp_for_repo.
  - ingest: [standalone/src/service/graph_service.rs](standalone/src/service/graph_service.rs)
    - call_mcp_mocks(repo_url, username, pat, sync=false)
  - sync_async: [standalone/src/handlers/ingest.rs](standalone/src/handlers/ingest.rs)
    - call_mcp_mocks(repo_url, username, pat, sync=true)
  - Parameters from standalone that affect it:
    - repo_url (ProcessBody.repo_url)
    - username (ProcessBody.username)
    - pat (ProcessBody.pat)
    - mocks (ProcessBody.mocks; used by should_call_mcp_for_repo to decide whether to call)

## Docs Agent
- Endpoint: POST /learn_docs
- Purpose: Summarizes rules/documentation files and stores a repository‑level documentation summary.
- Params:
  - repo_url (query, optional; limits to a specific repo)
- Responses:
  - 200: { message: "Documentation learned", summaries }
  - 200: { message: "Documentation already exists" }
  - 404: { error: "Repository not found" }
  - 500: { error: "Internal server error" }
- Standalone call path:
  - Triggered when ProcessBody.docs passes should_call_mcp_for_repo.
  - ingest: [standalone/src/service/graph_service.rs](standalone/src/service/graph_service.rs)
    - call_mcp_docs(repo_url, sync=false)
  - sync_async: [standalone/src/handlers/ingest.rs](standalone/src/handlers/ingest.rs)
    - call_mcp_docs(repo_url, sync=true)
  - Parameters from standalone that affect it:
    - repo_url (ProcessBody.repo_url)
    - docs (ProcessBody.docs; used by should_call_mcp_for_repo to decide whether to call)

## Describe Nodes Agent
- Endpoint: POST /repo/describe
- Purpose: Generates short descriptions for graph nodes missing docs and stores embeddings.
- Params (JSON body):
  - cost_limit (number, optional; default 0.5)
  - batch_size (integer, optional; default 25)
  - repo_url (string, optional; one or more repo URLs)
  - file_paths (string[], optional)
- Responses:
  - 200: { request_id, status: "pending", message }
  - 400: { error: "Invalid cost_limit..." } | { error: "Invalid batch_size..." } | { error: "No nodes found...", repo_paths }
- Standalone call path:
  - Triggered when ProcessBody.embeddings passes should_call_mcp_for_repo.
  - ingest: [standalone/src/service/graph_service.rs](standalone/src/service/graph_service.rs)
    - call_mcp_embed(repo_url, embeddings_limit, file_paths=[], sync=false)
  - sync: [standalone/src/service/graph_service.rs](standalone/src/service/graph_service.rs)
    - call_mcp_embed(repo_url, embeddings_limit, file_paths=modified_files, sync=true)
  - sync_async: [standalone/src/handlers/ingest.rs](standalone/src/handlers/ingest.rs)
    - call_mcp_embed(repo_url, embeddings_limit, file_paths=[], sync=true)
  - Parameters from standalone that affect it:
    - repo_url (ProcessBody.repo_url)
    - embeddings (ProcessBody.embeddings; used by should_call_mcp_for_repo to decide whether to call)
    - embeddings_limit (ProcessBody.embeddings_limit; maps to cost_limit)
