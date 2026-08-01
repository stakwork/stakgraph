# docker buildx build --platform linux/amd64,linux/arm64 -t repo2graph .
#
# NOTE: mcp depends on vein (`file:../vein`), which lives OUTSIDE this build
# context (`./mcp`). Stage vein's source into the context first so it can be
# COPY'd and built in-image:
#     yarn stage-vein   # (from mcp/) — copies ../vein into ./.vein
# CI does this in .github/workflows/publish-mcp.yml before the build.

# Use an official Node.js runtime as a parent image
FROM node:22-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies
RUN apt update && apt install -y sed git vim tree ripgrep wget curl jq python3 pandoc python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install openpyxl for generate_xlsx tool (--break-system-packages required under Debian bookworm PEP 668)
RUN pip install --break-system-packages openpyxl

# Install docx-mcp-server for stdio MCP Word document editing (tracked changes, comments, audit).
# A plain pip install places the `docx-mcp-server` console-script on PATH at
# /usr/local/bin/docx-mcp-server — callers should attach this server as:
#   { name: "docx", command: "docx-mcp-server" }
# Verified against v0.7.4: all 14 tools in the default filter exist in the package.
# NOTE: convert_to_pdf is deliberately excluded from the default tool filter —
# it requires LibreOffice/soffice which is NOT installed here to keep the image lean.
#
# Pin: docx-mcp-server 0.7.4 declares no upper bound on `mcp`, and mcp 2.0.0
# (2026-07-28) removed `mcp.server.fastmcp` (FastMCP -> MCPServer). Keep mcp on
# the 1.x line so docx-mcp's `from mcp.server.fastmcp import FastMCP` resolves.
# docx-mcp-server itself is pinned to the version this image is validated against,
# since it is otherwise fully unpinned and a future release that raises its own
# floor to mcp>=2.0.0 would make `mcp<2.0.0` unsatisfiable and turn this into a
# hard pip-resolver build failure instead of a controlled one.
# Both packages co-resolved in one invocation so the mcp<2.0.0 constraint is
# honoured up front (a separate later pip install could leave 2.0.0 in place).
RUN pip install --break-system-packages "docx-mcp-server==0.7.4" "mcp<2.0.0"

# Print resolved versions of key packages for a durable audit record.
# Uses pip freeze + grep so the step never fails due to an absent package name.
RUN pip freeze | grep -Ei 'mcp|httpx|pydantic|starlette|sse'

# Fail the build (not runtime) if the docx-mcp import chain doesn't load.
# Separate, unredirected steps so the real traceback surfaces in the build log.
RUN python -c "from mcp.server.fastmcp import FastMCP"

# Install GitHub CLI (gh) from the official apt repo (supports x64 and arm64)
RUN wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list && \
    apt update && apt install -y gh

# install Gitleaks (supports x64 and arm64)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/download/v8.28.0/gitleaks_8.28.0_linux_x64.tar.gz"; \
    elif [ "$ARCH" = "arm64" ]; then \
        GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/download/v8.28.0/gitleaks_8.28.0_linux_arm64.tar.gz"; \
    else \
        GITLEAKS_URL=""; \
    fi && \
    if [ -n "$GITLEAKS_URL" ]; then \
        echo "Installing gitleaks for architecture: $ARCH" && \
        wget -O /tmp/gitleaks.tar.gz "$GITLEAKS_URL" && \
        tar -xzf /tmp/gitleaks.tar.gz -C /tmp && \
        mv /tmp/gitleaks /usr/local/bin/gitleaks && \
        chmod +x /usr/local/bin/gitleaks && \
        rm /tmp/gitleaks.tar.gz; \
    else \
        echo "Skipping gitleaks install: unsupported architecture $ARCH"; \
    fi

# Install stakgraph CLI (bump version after new CLI release)
RUN curl -fsSL https://raw.githubusercontent.com/stakwork/stakgraph/refs/heads/main/install.sh | STAKGRAPH_VERSION=v0.4.25 STAKGRAPH_BIN_DIR=/usr/local/bin bash

RUN npx -y playwright install-deps
RUN npx -y playwright install

# Build vein (engine + web UI). mcp consumes it as `file:../vein` and serves
# its UI under /lab. `.vein` is vein's source staged into this context (see
# the header note / publish-mcp.yml). Built at /usr/src/vein so that
# `file:../vein` from /usr/src/app resolves to it.
#
# vein KEEPS its own runtime node_modules: Node resolves a package's imports
# from the real path of the file, so /usr/src/vein/build/index.js looks for its
# deps (zod, hono, ai, ...) under /usr/src/vein/node_modules — it never reaches
# /usr/src/app/node_modules. Sharing mcp's tree would also be wrong: vein needs
# zod@3 while mcp pins zod@4. We drop only devDependencies (npm prune) and the
# web build's node_modules (web is built to web/dist and served as static).
COPY ./.vein /usr/src/vein
# `npm install` (not `npm ci`): package-lock.json is gitignored repo-wide, so
# it isn't present in the staged `.vein` source — `npm ci` would fail. Matches
# the `npm install` used elsewhere in this Dockerfile (web, benchmark).
RUN cd /usr/src/vein && npm install && npm run build && \
    npm --prefix web install && npm --prefix web run build && \
    rm -rf web/node_modules && npm prune --omit=dev

# Copy package.json and package-lock.json (if available)
COPY ./package.json ./package.json
COPY ./yarn.lock ./yarn.lock

# Copy workspace package.json files (required for yarn workspaces)
COPY ./src/aieo/package.json ./src/aieo/package.json
COPY ./src/gitsee/agent/package.json ./src/gitsee/agent/package.json

# Install dependencies
RUN yarn

# Install linux x64 tokenizer bindings (required for fastembed in Docker)
RUN npm install --force --os=linux --cpu=x64 @anush008/tokenizers-linux-x64-gnu@0.0.0

# Copy the rest of your application code
COPY ./tsconfig.json ./tsconfig.json
COPY ./src ./src
COPY ./scripts ./scripts

# Build aieo first so gitsee-agent can find the dist folder
RUN cd src/aieo && yarn build

RUN mkdir -p ./docs
COPY ./docs/redoc-static.html ./docs/redoc-static.html

# Docgen assets (build_workbook.py + reference templates). Staged into src/repo
# here; `yarn build` (scripts/copy-lab-assets.mjs) copies them into build/repo/
# alongside the compiled docgen.js, which resolves them via import.meta.url.
COPY ./src/repo/build_workbook.py ./src/repo/build_workbook.py
COPY ./src/repo/docgen-templates ./src/repo/docgen-templates
COPY ./textarea ./textarea
COPY ./app ./app
RUN mkdir -p ./tests
COPY ./tests/playwright.config.js ./tests/playwright.config.js
COPY ./tests/tests.html ./tests/tests.html
COPY ./tests/staktrak/dist/staktrak.js ./tests/staktrak/dist/staktrak.js
COPY ./tests/staktrak/dist/playwright-generator.js ./tests/staktrak/dist/playwright-generator.js
COPY ./tests/frame/app.js ./tests/frame/app.js
COPY ./tests/frame/style.css ./tests/frame/style.css
COPY ./tests/frame/frame.html ./tests/frame/frame.html
COPY ./tests/app.js ./tests/app.js
COPY ./tests/style.css ./tests/style.css
COPY ./tests/hooks.js ./tests/hooks.js

# Build the application
RUN yarn build

# Build web SPA
COPY ./web ./web
RUN cd web && npm install && npm run build

# Build benchmark SPA
COPY ./benchmark ./benchmark
RUN cd benchmark && npm install && npm run build

# Install agent skills. Baked OUTSIDE /root and exposed via SKILLS_ROOT:
# sphinx-swarm mounts a named volume over /root, which is initialized from the
# image only once and then shadows /root/.agents/skills on every image update —
# skills shipped in newer images silently never appear (stale-volume bug).
RUN npx -y skills add https://github.com/anthropics/skills --skill frontend-design -y -g && \
    npx -y skills add https://github.com/getsentry/skills --skill code-simplifier -y -g && \
    npx -y skills add https://github.com/getsentry/skills --skill security-review -y -g && \
    mkdir -p /usr/src/skills && \
    cp -r /root/.agents/skills/. /usr/src/skills/ && \
    rm -rf /root/.agents/skills
ENV SKILLS_ROOT=/usr/src/skills

# Vendored skill packs (./skills -> SKILLS_ROOT). COPY merges into the existing
# directory, so the npx-installed skills above survive alongside them.
# A directory with a SKILL.md is a skill; a directory of skill directories is a
# pack contributing ONE line to the prompt index. Regenerate with
# `node scripts/vendor-skills.mjs`. Placed late: skill markdown changes far more
# often than the dep installs above it.
COPY ./skills /usr/src/skills

# Download fastembed models (~128MB) - only on amd64
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        echo "Downloading fastembed models for amd64..." && \
        node scripts/download-models.js; \
    else \
        echo "Skipping fastembed model download for $ARCH"; \
    fi

EXPOSE 3000

WORKDIR /usr/src/app

# Define the command to run your app.
# `--import tsx` registers a TypeScript loader so vein can dynamically import
# `.ts` custom steps at runtime (the lab seeds steps as TS source).
CMD ["node", "--import", "tsx", "build/index.js"]
