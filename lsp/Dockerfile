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
    ruby \
    ruby-dev \
    libyaml-dev \
    build-essential \
    automake \
    gcc \
    g++ \
    sed \
    unzip \
    default-jdk \
    gradle \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup.sh \
    && chmod +x rustup.sh \
    && ./rustup.sh -y

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

# Install gopls without CGO
RUN mkdir -p $GOPATH/bin \
    && CGO_ENABLED=0 go install -v golang.org/x/tools/gopls@v0.16.2

# ruby
RUN gem install ruby-lsp

# kotlin - LSP
RUN mkdir -p /opt/kotlin-language-server && \
    curl -L "https://github.com/fwcd/kotlin-language-server/releases/latest/download/server.zip" -o /opt/kotlin-language-server/server.zip && \
    cd /opt/kotlin-language-server && \
    unzip server.zip && \
    chmod +x server/bin/kotlin-language-server && \
    mkdir -p server/plugins && \
    curl -L "https://github.com/fwcd/kotlin-language-server/releases/latest/download/kotlin-analysis-plugin.jar" -o server/plugins/kotlin-analysis-plugin.jar 2>/dev/null || echo "Analysis plugin not found, skipping" && \
    curl -L "https://github.com/fwcd/kotlin-language-server/releases/latest/download/kotlin-structure-plugin.jar" -o server/plugins/kotlin-structure-plugin.jar 2>/dev/null || echo "Structure plugin not found, skipping" && \
    rm server.zip

# Kotlin compiler for improved language features and more accurate AST
RUN LATEST_KOTLIN_VERSION=$(curl -s https://api.github.com/repos/JetBrains/kotlin/releases/latest | jq -r '.tag_name' | sed 's/^v//') && \
    echo "Installing Kotlin compiler version: $LATEST_KOTLIN_VERSION" && \
    curl -L "https://github.com/JetBrains/kotlin/releases/download/v${LATEST_KOTLIN_VERSION}/kotlin-compiler-${LATEST_KOTLIN_VERSION}.zip" -o /tmp/kotlin-compiler.zip && \
    mkdir -p /opt/kotlin && \
    unzip /tmp/kotlin-compiler.zip -d /opt/kotlin && \
    rm /tmp/kotlin-compiler.zip && \
    chmod +x /opt/kotlin/kotlinc/bin/kotlin && \
    chmod +x /opt/kotlin/kotlinc/bin/kotlinc

# SLF4J implementation to fix logging warnings
RUN curl -L "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.36/slf4j-simple-1.7.36.jar" \
    -o /opt/kotlin-language-server/server/lib/slf4j-simple-1.7.36.jar

# Set environment variables for enhanced Kotlin support
ENV PATH="/opt/kotlin-language-server/server/bin:/opt/kotlin/kotlinc/bin:$PATH"
ENV KOTLIN_LANGUAGE_SERVER_OPTS="-Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -XX:+UseStringDeduplication"
ENV KOTLIN_COMPILER_HOME="/opt/kotlin/kotlinc"

# rust-analyzer
RUN curl -LO "https://github.com/rust-lang/rust-analyzer/releases/download/2025-01-20/rust-analyzer-x86_64-unknown-linux-gnu.gz" \
    && gzip -cd rust-analyzer-x86_64-unknown-linux-gnu.gz > /bin/rust-analyzer \
    && chmod +x /bin/rust-analyzer