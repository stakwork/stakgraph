# docker buildx build --platform linux/amd64,linux/arm64 -t repo2graph .

# Use an official Node.js runtime as a parent image
FROM node:22-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies
RUN apt update && apt install -y sed git vim tree ripgrep wget curl jq python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

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

# Install stakgraph CLI
RUN curl -fsSL https://raw.githubusercontent.com/stakwork/stakgraph/refs/heads/main/install.sh | STAKGRAPH_BIN_DIR=/usr/local/bin bash

RUN npx -y playwright install-deps
RUN npx -y playwright install

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

# Define the command to run your app
CMD ["node", "build/index.js"]
