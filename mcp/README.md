### build docs

redocly build-docs docs/swagger.yaml --output docs/redoc-static.html

# try it out

### install LSP

##### typescript

`npm install -g typescript-language-server`

##### go

`go install golang.org/x/tools/gopls@latest`

### generate graph data (in stakgraph top level dir)

export REPO_URL="https://github.com/stakwork/sphinx-tribes.git,https://github.com/stakwork/sphinx-tribes-frontend.git"
export OUTPUT_FORMAT=jsonl
export OUTPUT_NAME=tribes
cargo run --bin index

### load data into neo4j (in this dir)

docker-compose -f neo4j.yaml up -d

curl -X POST \
 -F "nodes=@./ast/examples/tribes-nodes.jsonl" \
 -F "edges=@./ast/examples/tribes-edges.jsonl" \
 http://localhost:3355/upload
