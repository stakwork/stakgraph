### openapi

redocly build-docs docs/swagger.yaml --output docs/redoc-static.html

### dev`

yarn dev

### run the graph

docker-compose -f neo4j.yaml up -d

docker-compose -f neo4j.yaml down

### upload files

curl -X POST \
 -F "nodes=@./ast/examples/urls-nodes.jsonl" \
 -F "edges=@./ast/examples/urls-edges.jsonl" \
 http://localhost:3000/upload

curl -X POST \
 -F "nodes=@./ast/examples/tribes-nodes.jsonl" \
 -F "edges=@./ast/examples/tribes-edges.jsonl" \
 http://localhost:3000/upload

curl -X POST \
 -F "nodes=@./ast/examples/stak-nodes.jsonl" \
 -F "edges=@./ast/examples/stak-edges.jsonl" \
 http://localhost:3000/upload

curl -X POST \
 -F "nodes=@./ast/examples/demo-repo-nodes.jsonl" \
 -F "edges=@./ast/examples/demo-repo-edges.jsonl" \
 http://localhost:3000/upload

### demo repo test endpoints

http://localhost:3000/map?node_type=Function&name=NewPerson

### set up in cursor

Cursor Settings -> Features -> MCP Servers -> Add New

sse, http://localhost:3000/sse

Composer -> Agent Mode will have access to the tool

### docker

docker run --rm -p 3000:3000 -e NEO4J_HOST=host.docker.internal repo2graph

docker build -t repo2graph .

docker tag repo2graph sphinxlightning/repo2graph:latest

docker push sphinxlightning/repo2graph:latest

### build for swarm:

make sure you are in mcp dir

docker buildx build --platform linux/amd64 -t sphinxlightning/repo2graph:latest . --push

### get

curl "http://localhost:3000/feature_code?page_name=%2Fleaderboard"

[
'https://github.com/stakwork/sphinx-tribes/commit/22a866b12268a928d500530fc352fa26e4def1ff',
'https://github.com/stakwork/sphinx-tribes-frontend/commit/ecc64bd268df0b108fbef56f1eeb7c32a76d6717'
]

curl "https://mcp.repo2graph.sphinx.chat/feature_code?page_name=%2Fleaderboard"
