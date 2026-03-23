<p align="center">
  <img src="./mcp/docs/sg.png" alt="StakGraph" width="700">
</p>

<h3 align="center">Your AI agent wastes thousands of tokens reading files over and over.<br>Give it a code graph instead.</h3>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#cli">CLI</a> &bull;
  <a href="#mcp-server">MCP Server</a> &bull;
  <a href="#graph-server">Graph Server</a> &bull;
  <a href="#languages">Languages</a>
</p>

---

StakGraph parses source code into a graph of **functions, classes, endpoints, data models, tests**, and their relationships -- using [tree-sitter](https://tree-sitter.github.io/tree-sitter/), instantly, with zero config.

1. **CLI** -- install in 10 seconds, point at any file or directory
2. **MCP server** -- plug into Cursor, Claude Code, Windsurf, OpenCode
3. **Graph server** -- Neo4j-backed querying, embedding, and visualization

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/stakwork/stakgraph/refs/heads/main/install.sh | bash
```

Pre-built binaries for **Linux** (x86_64, aarch64), **macOS** (Intel, Apple Silicon), and **Windows**.

## CLI

Point `stakgraph` at any file or directory. It parses the code, extracts every meaningful entity, maps their call relationships, and prints a structured summary.

### Parse a file

```
$ stakgraph src/routes.ts
```

```ansi
[1;36mFile:[0m [36msrc/routes.ts[0m

[1;35mDatamodel[0m: PersonRequest [2m(6)[0m
  type PersonRequest = Request<{}, {}, { name: string; email: string }>;

[1;35mDatamodel[0m: ResponseStatus [2m(9-14)[0m
  enum ResponseStatus {
    SUCCESS = 200,
    CREATED = 201,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500,
  }

[1;33mEndpoint[0m: POST /people/new [2m(18)[0m
[1;33mEndpoint[0m: GET /people/recent [2m(19)[0m

[1;32mFunction[0m: registerRoutes [2m(21-30)[0m
  export function registerRoutes(app)
  [2m→[0m /person/:id [2m(L22)[0m
  [2m→[0m /person [2m(L24)[0m

[1;33mEndpoint[0m: GET /person/:id [2m(22)[0m
[1;33mEndpoint[0m: POST /person [2m(24)[0m

[1;32mFunction[0m: getPerson [2m(32-49)[0m
  async function getPerson(req: Request, res: Response)

[1;32mFunction[0m: createPerson [2m(54-65)[0m
  [2mDocs:[0m Create a new person.
  async function createPerson(req: PersonRequest, res: PersonResponse)
```

Functions, endpoints, data models, classes, traits, tests -- all extracted with **line numbers**, **doc comments**, **signatures**, and **call edges** (`→`).

### Parse a directory

Parse multiple files at once. Cross-file function calls are resolved automatically:

```
$ stakgraph src/routes.py src/models.py src/db.py
```

```ansi
[1;36mFile:[0m [36msrc/routes.py[0m

[1;33mEndpoint[0m: GET /person/{id} [2m(10)[0m

[1;32mFunction[0m: get_person [2m(11-25)[0m
  [2mDocs:[0m Get user details by user id
  async def get_person(id: int, db: Session = Depends(get_db))
  [2m→[0m get_person_by_id [2m(L32)[0m [2m[db.py][0m

[1;33mEndpoint[0m: POST /person/ [2m(29)[0m
  [2mDocs:[0m Create new user endpoint

[1;32mFunction[0m: create_person [2m(30-41)[0m
  [2mDocs:[0m Create new user
  async def create_person(person: CreateOrEditPerson, db: Session = Depends(get_db))
  [2m→[0m create_new_person [2m(L37)[0m [2m[db.py][0m

[1;36mFile:[0m [36msrc/models.py[0m

[1;34mClass[0m: Person [2m(9-27)[0m
  [2mDocs:[0m Person model for storing user details

[1;35mDatamodel[0m: CreateOrEditPerson [2m(30-36)[0m
  [2mDocs:[0m PersonCreate model for creating new person
  class CreateOrEditPerson(BaseModel):
      id: Optional[int] = None
      name: str
      email: str

[1;36mFile:[0m [36msrc/db.py[0m

[1;32mFunction[0m: get_person_by_id [2m(32-34)[0m
  [2mDocs:[0m Get a person by their ID

[1;32mFunction[0m: create_new_person [2m(37-56)[0m
  [2mDocs:[0m Create a new person in the database
```

Notice `→ get_person_by_id (L32) [db.py]` -- cross-file call resolution with the exact target line and filename.

### Summarize a project

Get a token-budget-aware overview of any project. Prioritizes entry points and key files:

```
$ stakgraph summarize ./my-project --max-tokens 2000
```

```ansi
[1mSummary:[0m [1;36m./my-project[0m  (budget: [1;33m2000[0m tokens)

[1;4mDirectory Structure[0m
[1;36mmy-project/[0m
  [1msrc/[0m
    middleware/
    routers/
    services/
    index.ts
    model.ts
    routes.ts
    service.ts

[1;4mFile Summaries[0m
[1;36mFile:[0m [36msrc/routes.ts[0m
[1;33mEndpoint[0m: POST /people/new [2m(18)[0m
[1;33mEndpoint[0m: GET /people/recent [2m(19)[0m
[1;32mFunction[0m: registerRoutes [2m(21-30)[0m
  [2m→[0m /person/:id [2m(L22)[0m
  [2m→[0m /person [2m(L24)[0m
[1;32mFunction[0m: getPerson [2m(32-49)[0m
[1;32mFunction[0m: createPerson [2m(54-65)[0m

[1;36mFile:[0m [36msrc/services/user-service.ts[0m
[1;34mClass[0m: UserService [2m(3-24)[0m
[1;32mFunction[0m: UserService.findAll [2m(4-7)[0m
[1;32mFunction[0m: UserService.findById [2m(9-11)[0m
[1;32mFunction[0m: UserService.create [2m(13-15)[0m

[2m[1418/2000 tokens used — 11 files not shown][0m
```

Designed to fit into LLM context windows. Adaptive depth, file scoring (entry points first), and token counting via tiktoken.

### Track code changes

See what actually changed in your code at the structural level -- not just line diffs, but which **functions, endpoints, and classes** were added, removed, or modified:

```
$ stakgraph changes diff --since main src/
```

```ansi
[1;36mFound[0m [1;32m9[0m file(s) changed in [33mchanges since main[0m (scope: [36msrc/[0m)

[1msrc/args.rs[0m  [1;33m[Modified][0m  [2m(3 nodes)[0m
  [1;33m~[0m [33mClass[0m [1mCliArgs[0m  [2mL8-L51[0m
  [1;33m~[0m [33mDatamodel[0m [1mCliArgs[0m  [2mL8-L51[0m

[1msrc/changes.rs[0m  [1;33m[Modified][0m  [2m(4 nodes)[0m
  [1;33m~[0m [33mFunction[0m [1mrun[0m  [2mL19-L41[0m
    [31m-[0m [31mpub async fn run(args: &ChangesArgs, out: &mut Output) -> Result<()> {[0m
    [32m+[0m [32mpub async fn run(args: &ChangesArgs, out: &mut Output, show_progress: bool) -> Result<()> {[0m

[1msrc/git.rs[0m  [1;32m[Added][0m  [2m(1 node)[0m
  [1;32m+[0m [32mFunction[0m [1mget_repo_root[0m  [2mL2[0m

[1msrc/progress.rs[0m  [1;33m[Modified][0m  [2m(6 nodes)[0m
  [1;33m~[0m [33mFunction[0m [1mnew[0m  [2mL12-L27[0m
    [31m-[0m [31mpub fn new(verbose: bool) -> (Self, broadcast::Sender<StatusUpdate>) {[0m
    [32m+[0m [32mpub fn new(message: &str) -> Self {[0m
  [1;32m+[0m [32mDatamodel[0m [1mCliSpinner[0m  [2mL7[0m
  [1;32m+[0m [32mFunction[0m [1mset_message[0m  [2mL29[0m
  [1;32m+[0m [32mFunction[0m [1mfinish_and_clear[0m  [2mL35[0m
  [1;32m+[0m [32mFunction[0m [1mfinish_with_message[0m  [2mL41[0m

[1m9 files[0m  [32m10 added[0m  [31m0 removed[0m  [33m13 modified[0m
```

Works with `--staged`, `--last N`, `--since <ref>`, or `--range HEAD~5..HEAD`. Builds before/after AST graphs from git blobs and computes the structural delta.

### CLI options

```
stakgraph <files/dirs>                   # parse and print graph summary
stakgraph <files/dirs> --allow           # include unverified function calls
stakgraph <files/dirs> --skip-calls      # skip call graph extraction
stakgraph <files/dirs> --no-nested       # exclude nested nodes

stakgraph summarize <dir>               # token-budget project summary
stakgraph summarize <dir> --max-tokens N # set token budget (default: 5000)

stakgraph changes list <paths>           # list commits touching paths
stakgraph changes diff --staged          # graph diff of staged changes
stakgraph changes diff --last 3          # diff last 3 commits
stakgraph changes diff --since main      # diff since branch point
stakgraph changes diff --range a..b      # diff between two refs
```

---

## What it extracts

StakGraph understands the semantic structure of code, not just syntax:

| Node Type     | Examples                                              |
| ------------- | ----------------------------------------------------- |
| **Function**  | Functions, methods, handlers, callbacks               |
| **Endpoint**  | HTTP routes (`GET /users`, `POST /api/v1/login`)      |
| **Request**   | HTTP client calls to external services                |
| **DataModel** | Structs, interfaces, types, enums, schemas            |
| **Class**     | Classes with method ownership                         |
| **Trait**     | Interfaces, abstract classes, protocols               |
| **Test**      | Unit tests, integration tests, E2E tests (classified) |
| **Import**    | Module imports with resolution                        |

And the relationships between them:

| Edge Type      | Meaning                             |
| -------------- | ----------------------------------- |
| **Calls**      | Function A calls Function B         |
| **Handler**    | Endpoint handled by Function        |
| **Contains**   | File/Module contains Function/Class |
| **Operand**    | Class owns Method                   |
| **Implements** | Class implements Trait              |
| **ParentOf**   | Class inheritance                   |

---

## Languages

16 languages with framework-aware parsing:

| Language       | Frameworks                |
| -------------- | ------------------------- |
| **TypeScript** | React, Express, Nest.js   |
| **JavaScript** | React, Express            |
| **Python**     | FastAPI, Django, Flask    |
| **Go**         | Gin, Echo, net/http       |
| **Rust**       | Axum, Actix, Rocket       |
| **Ruby**       | Rails (routes, ERB, HAML) |
| **Java**       | Spring Boot               |
| **Kotlin**     | Spring, Ktor              |
| **Swift**      | Vapor                     |
| **C#**         | ASP.NET                   |
| **PHP**        | Laravel                   |
| **C / C++**    |                           |
| **Angular**    | Components, services      |
| **Svelte**     | Components                |
| **Bash**       |                           |
| **TOML**       | Config parsing            |

---

## MCP Server

The MCP server exposes StakGraph's graph intelligence to AI agents running in Cursor, Claude Code, Windsurf, OpenCode, or any MCP-compatible editor.

### Tools

| Tool                      | What it does                                                                    |
| ------------------------- | ------------------------------------------------------------------------------- |
| `stakgraph_search`        | Fulltext or vector (semantic) search across the codebase graph                  |
| `stakgraph_get_nodes`     | Retrieve nodes by type, name, or file                                           |
| `stakgraph_get_edges`     | Retrieve edges connecting nodes                                                 |
| `stakgraph_map`           | Visual map of code relationships from any node (configurable depth & direction) |
| `stakgraph_repo_map`      | Directory/file tree of the repository                                           |
| `stakgraph_code`          | Retrieve actual code from a subtree                                             |
| `stakgraph_shortest_path` | Find shortest path between two nodes in the graph                               |
| `stakgraph_explore`       | Autonomous AI agent that explores the codebase and answers questions            |

### Built-in agents

- **Explore Agent** -- AI-driven codebase exploration using the "zoom pattern" (Overview → Files → Functions → Dependencies). Configurable LLM provider (Anthropic, OpenAI, Google, OpenRouter).
- **Describe Agent** -- Generates descriptions for undocumented nodes and stores embeddings for semantic search.
- **Docs Agent** -- Summarizes documentation and rules files across the repo.
- **Mocks Agent** -- Scans for 3rd-party service integrations and records mock coverage.

### Gitree: Feature knowledge from git history

Gitree extracts feature-level knowledge from PR and commit history using LLM analysis:

```bash
yarn gitree process <owner> <repo>     # extract features from PR history
yarn gitree summarize-all              # generate docs for all features
yarn gitree search-clues "auth flow"   # semantic search across architectural clues
```

Builds a knowledge base of **Features**, **PRs**, **Commits**, and **Clues** (architectural insights like patterns, conventions, gotchas, data flows). Links them to code entities in the graph.

---

## Graph Server

For full-scale codebase indexing, StakGraph runs as an HTTP server backed by **Neo4j**:

```bash
docker-compose up    # starts Neo4j + StakGraph server on port 7799
```

### Ingest repositories

```bash
# Parse one or more repos into the graph
export REPO_URL="https://github.com/org/backend.git,https://github.com/org/frontend.git"
cargo run --bin index
```

Endpoints and requests are linked across repos -- a `POST /api/users` endpoint in the backend connects to the `fetch("/api/users")` request in the frontend.

### Query the graph

<img src="./mcp/docs/neo4j_screenshot.png" alt="Neo4j Graph" width="700">

The graph stores 21 node types and 13 edge types. Query with Cypher, search with fulltext or vector similarity, or use the MCP tools.

### Vector search

Code is embedded using **BGE-Small-EN-v1.5** (384 dimensions) via fastembed. Weighted pooling prioritizes function signatures. Search semantically across the entire codebase:

```
POST /search
{ "query": "user authentication middleware", "limit": 10 }
```

### API endpoints

| Endpoint              | Description                             |
| --------------------- | --------------------------------------- |
| `POST /process`       | Parse and index a repository            |
| `POST /embed_code`    | Generate embeddings for code            |
| `GET /search`         | Fulltext or vector search               |
| `GET /map`            | Relationship map from a node            |
| `GET /shortest_path`  | Path between two nodes                  |
| `GET /tests/coverage` | Test coverage analysis                  |
| `POST /ingest_async`  | Background repo ingestion with webhooks |

---

## Architecture

```
stakgraph/
├── ast/          # Core Rust library: tree-sitter parsing → graph of nodes & edges
├── cli/          # CLI binary: parse, summarize, diff
├── lsp/          # LSP integration for precise symbol resolution
├── standalone/   # Axum HTTP server wrapping the ast library
├── mcp/          # TypeScript MCP server with agents, gitree, vector search
└── shared/       # Shared types
```

The `ast` crate is the engine. It takes source files, runs tree-sitter queries to extract nodes, resolves cross-file calls (optionally via LSP), and produces a graph. The graph can be:

- **In-memory** (`ArrayGraph`) -- used by the CLI, fast, no dependencies
- **Neo4j** (`Neo4jGraph`) -- persistent, queryable, used by the server

---

## Contributing

```bash
cargo test                # run tests
USE_LSP=1 cargo test      # run tests with LSP resolution
```

You may need to install LSPs:

```bash
# TypeScript
npm install -g typescript typescript-language-server

# Go
go install golang.org/x/tools/gopls@latest

# Rust
rustup component add rust-analyzer

# Python
pip install python-lsp-server
```

---

<p align="center">
  <a href="https://github.com/stakwork/stakgraph">github.com/stakwork/stakgraph</a>
</p>
