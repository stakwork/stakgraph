# docker buildx build --platform linux/amd64,linux/arm64 -t repo2graph .
#
# NOTE: mcp depends on strut as a git dependency (see package.json). `yarn`
# below clones github.com/stakwork/strut at the pinned commit and runs its
# `prepare` (engine tsc + web UI build) in-image, so the build needs network
# access to github.com and `git` (installed below). Nothing to stage.

# Use an official Node.js runtime as a parent image
FROM node:22-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies
RUN apt update && apt install -y sed git git-lfs vim tree ripgrep wget curl jq python3 pandoc python3-pip python3-venv zip unzip poppler-utils ffmpeg bzip2 && \
    git lfs install --system && \
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

# Agent python venv — the python that agent-step `bash` (and the lab's GAIA
# scorer) sees. A SEPARATE venv, not the system env: the system python carries
# the pinned docx-mcp/mcp<2.0.0 resolution above, and mixing data-science deps
# into it risks resolver conflicts on future builds; the venv also can't be
# broken by a future docx-mcp pin change. Prepending its bin to PATH makes
# `python3`/`python` resolve here for agent bash and for docgen's
# build_workbook.py (openpyxl included for exactly that); docx-mcp-server is
# unaffected (its console script pins the system interpreter via shebang).
# Contents: PDF text extraction (pypdf, pdfplumber; pdftotext comes from
# poppler-utils), spreadsheets (openpyxl, pandas), numerics (numpy — also
# needed by the GAIA scorer), images (pillow), scraping (requests, bs4), and
# yt-dlp (with ffmpeg from apt) for media tasks.
RUN python3 -m venv /usr/src/agent-venv && \
    /usr/src/agent-venv/bin/pip install --no-cache-dir \
      numpy pandas openpyxl pypdf pdfplumber pillow requests beautifulsoup4 yt-dlp && \
    /usr/src/agent-venv/bin/python -c "import numpy, pandas, openpyxl, pypdf, pdfplumber, PIL, requests, bs4"
ENV PATH="/usr/src/agent-venv/bin:$PATH"

# NOTE (lab / GAIA): the dataset is deliberately NOT baked into the image —
# GAIA's terms forbid resharing it outside a gated/private repo, and a
# published image is neither. It is fetched at runtime by
# src/lab/gaia/bootstrap.ts.
#
# HF_TOKEN is REQUIRED at runtime: the dataset is gated and the fetch is
# refused without credentials (anonymous `git ls-remote` succeeding against
# the repo is a red herring — `git-upload-pack` is what's gated). Supply a
# READ token from an account that has accepted GAIA's terms; no username is
# needed (HF takes the token as the password with any username). Only a
# pre-populated checkout mounted at GAIA_DIR removes the need for it.
#
# GAIA_PYTHON is unnecessary here: the agent venv above is first on PATH and
# carries the numpy the scorer needs. GAIA_SCORER_SHA256 is unnecessary too —
# the pin lives in the repo (bootstrap.ts SCORER_SHA256).
#
# git-lfs (installed above) is REQUIRED: GAIA's attachments are LFS-backed and
# a clone without it silently yields ~130-byte pointer stubs where the xlsx /
# pdf / mp3 files should be.
#
# Mount a volume at the cache dir so the ~205MB clone happens once rather than
# per container:  -v gaia-cache:/root/.cache/strut  (override with STRUT_CACHE_DIR
# or pin the checkout outright with GAIA_DIR).

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

# Copy package.json and package-lock.json (if available)
COPY ./package.json ./package.json
COPY ./yarn.lock ./yarn.lock
# .yarnrc sets network-concurrency 1; without it the `yarn` below dies on a
# cold cache (see the comment in .yarnrc).
COPY ./.yarnrc ./.yarnrc

# Copy workspace package.json files (required for yarn workspaces)
COPY ./src/aieo/package.json ./src/aieo/package.json
COPY ./src/gitsee/agent/package.json ./src/gitsee/agent/package.json

# Install dependencies. This also clones strut (git dep in package.json) and
# runs its `prepare`, which builds the engine and the /lab web UI; the packed
# result lands in node_modules/strut with build/ and web/dist. strut's own
# deps are resolved into this tree, so the `resolutions` above apply to it.
RUN yarn

# Install linux x64 tokenizer bindings (required for fastembed in Docker)
#
# Fetched with `npm pack` + tar rather than `npm install`: an `npm install`
# on top of the yarn tree makes npm re-resolve all of package.json, and with
# strut as a git dependency npm would re-clone and re-build strut its own way
# (and did fail doing so: the `--os/--cpu` flags leak into that nested build
# and vite gets the wrong rollup binary on arm64). `npm pack` only downloads
# the one tarball, so node_modules stays exactly what yarn produced.
#
# The onnxruntime-node check below stays: yarn's `resolutions` pin it to one
# copy (strut's binding and fastembed's must share one libonnxruntime.so, same
# soname), and the build should fail, not /lab at runtime, if a second copy
# ever sneaks in.
RUN mkdir -p node_modules/@anush008/tokenizers-linux-x64-gnu && \
    npm pack @anush008/tokenizers-linux-x64-gnu@0.0.0 --pack-destination /tmp >/dev/null && \
    tar xzf /tmp/anush008-tokenizers-linux-x64-gnu-0.0.0.tgz --strip-components=1 -C node_modules/@anush008/tokenizers-linux-x64-gnu && \
    rm /tmp/anush008-tokenizers-linux-x64-gnu-0.0.0.tgz
RUN extra="$(find node_modules -mindepth 2 -type d -name onnxruntime-node -not -path 'node_modules/onnxruntime-node' -not -path 'node_modules/strut/*')"; \
    if [ -n "$extra" ]; then echo "duplicate onnxruntime-node copies (soname clash with strut):"; echo "$extra"; exit 1; fi

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
# `--import tsx` registers a TypeScript loader so strut can dynamically import
# `.ts` custom steps at runtime (the lab seeds steps as TS source).
CMD ["node", "--import", "tsx", "build/index.js"]
