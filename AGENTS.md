# stakgraph

A source code parser using tree-sitter, LSP, and neo4j. Software knowledge graphs for AI agents.

## Sub-projects

### ast

Core Rust library that parses source code into a graph of nodes and edges using tree-sitter grammars. Supports 15+ languages including Go, TypeScript, Python, Rust, Ruby, Swift, Kotlin, Java, C/C++, and more.

### mcp

TypeScript MCP (Model Context Protocol) server that exposes the stakgraph tools to AI agents. Includes graph querying, repo ingestion, and vector search capabilities.

### standalone

Rust HTTP server (Axum) that wraps the `ast` library, providing a web API and webhook interface for parsing and indexing repositories.
