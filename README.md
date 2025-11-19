# stakgraph

A source code parser using treesitter, LSP, and neo4j. Software knowledge graphs for AI agents.

![img](./mcp/docs/sg.png)

### cli

```
curl -fsSL https://raw.githubusercontent.com/stakwork/stakgraph/refs/heads/main/install.sh | bash
```

```
stakgraph ./file.ts
```

### parse a repo

Example of parsing [sphinx-tribes](https://github.com/stakwork/demo-repo) and [sphinx-tribes-frontend](https://github.com/stakwork/sphinx-tribes-frontend). Endpoints, Requests, and E2E tests are linked between the two repos.

```bash
export REPO_URL="https://github.com/stakwork/sphinx-tribes.git,https://github.com/stakwork/sphinx-tribes-frontend.git"
cargo run --bin index
```

[ingest some data](https://github.com/stakwork/stakgraph/wiki/Ingest-some-data)

### Language support

- [x] Golang
- [x] React
- [x] Ruby on Rails
- [x] Typescript
- [x] Python
- [x] Swift
- [x] Kotlin
- [x] Rust
- [x] Java
- [x] Angular
- [x] Svelte
- [ ] C
- [ ] C#

### contributing

To verify locally that all tests pass before making a PR:

- `cargo test`
- `USE_LSP=1 cargo test`

You may need to install LSPs first:

### LSP

This project uses the Language Server Protocol to connect nodes in the graph.

##### typescript

`npm install -g typescript`

`npm install -g typescript-language-server`

##### go

`go install golang.org/x/tools/gopls@latest`

##### rust

`rustup component add rust-analyzer`

##### python

`pip install python-lsp-server`
