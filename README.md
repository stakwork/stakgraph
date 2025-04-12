# stakgraph

A source code parser using treesitter, LSP, and neo4j. Software knowledge graphs for AI agents.

![img](./mcp/docs/sg.png)

### parse a repo

Example of parsing [sphinx-tribes](https://github.com/stakwork/demo-repo) and [sphinx-tribes-frontend](https://github.com/stakwork/sphinx-tribes-frontend). Endpoints, Requests, and E2E tests are linked between the two repos.

```bash
export REPO_URL="https://github.com/stakwork/sphinx-tribes.git,https://github.com/stakwork/sphinx-tribes-frontend.git"
cargo run --bin index
```

[ingest some data](https://github.com/stakwork/stakgraph/wiki/Ingest-some-data)

### language support

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

### LSP

This project uses the Language Server Protocol to connect nodes in the graph.

##### go

`go install golang.org/x/tools/gopls@latest`

##### rust

https://github.com/rust-lang/rust-analyzer/releases

##### typescript

`npm install -g typescript-language-server`

##### python

`pip install python-lsp-server`
