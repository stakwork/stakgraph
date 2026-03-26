# CLI Examples

## task_manager

A small Rust project with two commits, designed to demonstrate all `stakgraph` CLI commands.

### Prerequisites

Build the CLI first (from the repo root):

```bash
cargo build --bin stakgraph
```

### Running the examples

All scripts must be run from inside `task_manager/`.

```bash
cd cli/examples/task_manager

./run_parse.sh    # parse source tree, single file, stats, named node lookup
./run_deps.sh     # BFS dependency tree from an entry point
./run_changes.sh  # commit history + graph delta for last commit
```

`run_parse.sh` and `run_deps.sh` work directly against the `src/` tree here.

`run_changes.sh` clones [`fayekelmith/task-manager-example`](https://github.com/fayekelmith/task-manager-example) into a temp directory (it needs a real git history), runs the demo, then cleans up. The repo has a stable two-commit history that will never change.
