FROM debian:bookworm

# deps
RUN apt-get update && apt-get install -y \
    ca-certificates \
    openssl \
    libssl-dev \
    pkg-config \
    curl \
    git \
    npm \
    nodejs \
    python3-full \
    python3-pip \
    libyaml-dev \
    # ruby build dependencies for compiling arbitrary ruby versions
    libreadline-dev \
    zlib1g-dev \
    libffi-dev \
    libgdbm-dev \
    libncurses5-dev \
    libxml2-dev \
    libxslt1-dev \
    liblzma-dev \
    default-jdk \
    openjdk-17-jre-headless \
    unzip \
    build-essential \
    automake \
    gcc \
    g++ \
    sed \
    bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Detect architecture and set JAVA_HOME accordingly
RUN echo "Verifying Java installation..." && \
    java -version && \
    mkdir -p /usr/lib/jvm && \
    JAVA_PATH="$(readlink -f /usr/bin/java | sed 's:/bin/java::')" && \
    echo "Detected Java path: $JAVA_PATH" && \
    if [ ! -d "$JAVA_PATH" ]; then \
        echo "Error: Could not detect Java installation directory"; \
        exit 1; \
    fi && \
    ln -sfT "$JAVA_PATH" /usr/lib/jvm/java-home

# Set and persist JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-home
ENV PATH="$JAVA_HOME/bin:$PATH"

# Sanity check
RUN java -version && \
    echo "JAVA_HOME = $JAVA_HOME" && \
    test -x "$JAVA_HOME/bin/java"

# rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup.sh \
    && chmod +x rustup.sh \
    && ./rustup.sh -y \
    && rm rustup.sh

# python
RUN pip install python-lsp-server --break-system-packages

# js
RUN npm install -g typescript typescript-language-server

# go
WORKDIR /tmp
RUN curl -O https://dl.google.com/go/go1.23.2.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz \
    && rm go1.23.2.linux-amd64.tar.gz

ENV GOROOT=/usr/local/go
ENV GOPATH=/root/go
ENV PATH=$PATH:$GOROOT/bin:$GOPATH/bin

RUN mkdir -p $GOPATH/bin \
    && CGO_ENABLED=0 go install -v golang.org/x/tools/gopls@v0.16.2

# Install asdf + Ruby 3.4.2 + Ruby LSPs
SHELL ["/bin/bash", "-lc"]
RUN set -eux; \
    export ASDF_DIR=/root/.asdf; \
    git clone https://github.com/asdf-vm/asdf.git "$ASDF_DIR" --branch v0.12.0; \
    echo '. $ASDF_DIR/asdf.sh' >> /etc/profile.d/asdf.sh; \
    . "$ASDF_DIR/asdf.sh"; \
    asdf plugin-add ruby https://github.com/asdf-vm/asdf-ruby.git || true; \
    asdf install ruby 3.4.2; \
    asdf global ruby 3.4.2; \
    gem install --no-document bundler -v 2.6.2 ruby-lsp solargraph

# Add asdf shims to PATH globally
ENV PATH="/root/.asdf/shims:/root/.asdf/bin:$PATH"

# rust-analyzer
SHELL ["/bin/sh", "-c"]
RUN curl -LO "https://github.com/rust-lang/rust-analyzer/releases/download/2025-01-20/rust-analyzer-x86_64-unknown-linux-gnu.gz" \
    && gzip -cd rust-analyzer-x86_64-unknown-linux-gnu.gz > /bin/rust-analyzer \
    && chmod +x /bin/rust-analyzer

# Kotlin Language Server
RUN set -eux; \
    echo "Verifying Java setup for Kotlin LSP..."; \
    java -version; \
    echo "JAVA_HOME = $JAVA_HOME"; \
    mkdir -p /opt/kotlin-language-server /tmp/kls; \
    curl -fSL --create-dirs -o /tmp/kls/kls.zip "https://github.com/fwcd/kotlin-language-server/releases/latest/download/server.zip"; \
    unzip /tmp/kls/kls.zip -d /tmp/kls; \
    if [ -d /tmp/kls/server ]; then \
        cp -r /tmp/kls/server/* /opt/kotlin-language-server/; \
    else \
        echo "ERROR: /tmp/kls/server directory not found after unzip"; \
        ls -la /tmp/kls; \
        exit 1; \
    fi; \
    if [ -f /opt/kotlin-language-server/bin/kotlin-language-server ]; then \
        ln -sf /opt/kotlin-language-server/bin/kotlin-language-server /usr/local/bin/kotlin-language-server; \
    elif [ -f /opt/kotlin-language-server/kotlin-language-server ]; then \
        ln -sf /opt/kotlin-language-server/kotlin-language-server /usr/local/bin/kotlin-language-server; \
    else \
        echo "ERROR: kotlin-language-server not found"; \
        ls -la /opt/kotlin-language-server; \
        exit 1; \
    fi; \
    chmod +x /usr/local/bin/kotlin-language-server; \
    rm -rf /tmp/kls
