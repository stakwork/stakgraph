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

# rust-analyzer
RUN curl -LO "https://github.com/rust-lang/rust-analyzer/releases/download/2025-01-20/rust-analyzer-x86_64-unknown-linux-gnu.gz" \
    && gzip -cd rust-analyzer-x86_64-unknown-linux-gnu.gz > /bin/rust-analyzer \
    && chmod +x /bin/rust-analyzer

# kotlin-lsp
RUN mkdir -p /opt/kotlin-language-server && \
    curl -L "https://github.com/fwcd/kotlin-language-server/releases/latest/download/server.zip" -o /opt/kotlin-language-server/server.zip && \
    cd /opt/kotlin-language-server && \
    unzip server.zip && \
    rm server.zip && \
    chmod +x server/bin/kotlin-language-server

ENV PATH="/opt/kotlin-language-server/server/bin:$PATH"