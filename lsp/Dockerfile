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
    default-jdk \
    unzip \
    build-essential \
    automake \
    gcc \
    g++ \
    sed \
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

# kotlin-language-server (robust)

RUN set -eux; \
    mkdir -p /opt/kotlin-language-server /tmp/kls; \
    echo "Downloading kotlin-language-server..."; \
    curl -fSL --create-dirs -o /tmp/kls/kls.zip "https://github.com/fwcd/kotlin-language-server/releases/latest/download/server.zip"; \
    unzip /tmp/kls/kls.zip -d /tmp/kls; \
    # Copy the contents
    if [ -d /tmp/kls/server ]; then \
        cp -r /tmp/kls/server/* /opt/kotlin-language-server/; \
    else \
        echo "ERROR: /tmp/kls/server directory not found after unzip"; \
        ls -la /tmp/kls; \
        exit 1; \
    fi; \
    # create a symlink for the expected executable name; fail if not found
    if [ -f /opt/kotlin-language-server/bin/kotlin-language-server ]; then \
        ln -sf /opt/kotlin-language-server/bin/kotlin-language-server /usr/local/bin/kotlin-language-server; \
    elif [ -f /opt/kotlin-language-server/kotlin-language-server ]; then \
        ln -sf /opt/kotlin-language-server/kotlin-language-server /usr/local/bin/kotlin-language-server; \
    else \
        echo "ERROR: kotlin-language-server executable not found in /opt/kotlin-language-server"; \
        echo "Listing /opt and /tmp/kls for debugging:"; ls -la /opt/kotlin-language-server || true; ls -la /tmp/kls || true; \
        exit 1; \
    fi; \
    chmod +x /usr/local/bin/kotlin-language-server; \
    rm -rf /tmp/kls