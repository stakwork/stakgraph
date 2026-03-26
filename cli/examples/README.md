# CLI Examples

## task_manager

A small Rust project with two commits, designed to demonstrate all `stakgraph` CLI commands.

### Prerequisites

Build the CLI first (from the repo root):

```bash
cargo build --bin stakgraph
```

### Running the examples

All scripts must be run from inside `task_manager/` (the `changes` commands require a `.git` there).

```bash
cd cli/examples/task_manager

./run_parse.sh    # parse source tree, single file, stats, named node lookup
./run_deps.sh     # BFS dependency tree from an entry point
./run_changes.sh  # commit history + graph delta for last commit
```
