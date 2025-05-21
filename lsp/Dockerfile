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
    sqlite3 \
    libsqlite3-dev \
    zlib1g-dev \
    libreadline-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install rbenv for Ruby version management
RUN git clone https://github.com/rbenv/rbenv.git ~/.rbenv \
    && cd ~/.rbenv && src/configure && make -C src \
    && echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc \
    && echo 'eval "$(rbenv init -)"' >> ~/.bashrc \
    && git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build

# Add rbenv to PATH
ENV PATH="/root/.rbenv/shims:/root/.rbenv/bin:${PATH}"

# Install latest stable Ruby as default
RUN rbenv install $(rbenv install -l | grep -v - | tail -1) \
    && rbenv global $(rbenv install -l | grep -v - | tail -1)

# Ruby LSP and essential gems
RUN gem install bundler --no-document \
    && gem install ruby-lsp ruby-lsp-rails ruby-lsp-rspec sorbet-runtime --no-document

# Create a smart Ruby LSP wrapper script
RUN echo '#!/bin/bash\n\
project_dir="$1"\n\
cd "$project_dir" || exit 1\n\
\n\
# Check for specific Ruby version\n\
if [ -f ".ruby-version" ]; then\n\
  ruby_version=$(cat .ruby-version | tr -d "\\n")\n\
  echo "Detected Ruby version: $ruby_version"\n\
  if ! rbenv versions | grep -q "$ruby_version"; then\n\
    echo "Installing Ruby $ruby_version..."\n\
    rbenv install "$ruby_version" || echo "Failed to install Ruby $ruby_version, using default"\n\
  fi\n\
  export RBENV_VERSION="$ruby_version"\n\
  echo "Using Ruby $ruby_version"\n\
fi\n\
\n\
# Handle project dependencies\n\
if [ -f "Gemfile" ]; then\n\
  # Check if bundler is installed for this Ruby version\n\
  if ! gem list -i bundler > /dev/null 2>&1; then\n\
    gem install bundler --no-document\n\
  fi\n\
  \n\
  # Install project gems\n\
  bundle config set --local path vendor/bundle 2>/dev/null || true\n\
  bundle install --quiet --jobs=4 --retry=3 || echo "Warning: bundle install failed, continuing anyway"\n\
  \n\
  # Install ruby-lsp gems if needed\n\
  ruby_lsp_needed=true\n\
  ruby_lsp_rails_needed=true\n\
  ruby_lsp_rspec_needed=true\n\
  \n\
  if bundle show ruby-lsp > /dev/null 2>&1; then\n\
    ruby_lsp_needed=false\n\
  fi\n\
  \n\
  if bundle show ruby-lsp-rails > /dev/null 2>&1; then\n\
    ruby_lsp_rails_needed=false\n\
  fi\n\
  \n\
  if bundle show ruby-lsp-rspec > /dev/null 2>&1; then\n\
    ruby_lsp_rspec_needed=false\n\
  fi\n\
  \n\
  # Check if this is a Rails project\n\
  is_rails=false\n\
  if grep -q "\\<rails\\>" Gemfile; then\n\
    is_rails=true\n\
  fi\n\
  \n\
  # Check if this is an RSpec project\n\
  is_rspec=false\n\
  if grep -q "\\<rspec\\>" Gemfile; then\n\
    is_rspec=true\n\
  fi\n\
  \n\
  # Inject necessary ruby-lsp gems if needed\n\
  if [ "$ruby_lsp_needed" = true ]; then\n\
    echo "Installing ruby-lsp..."\n\
    gem install ruby-lsp --no-document\n\
  fi\n\
  \n\
  if [ "$is_rails" = true ] && [ "$ruby_lsp_rails_needed" = true ]; then\n\
    echo "Installing ruby-lsp-rails..."\n\
    gem install ruby-lsp-rails --no-document\n\
  fi\n\
  \n\
  if [ "$is_rspec" = true ] && [ "$ruby_lsp_rspec_needed" = true ]; then\n\
    echo "Installing ruby-lsp-rspec..."\n\
    gem install ruby-lsp-rspec --no-document\n\
  fi\n\
  \n\
  # Use bundled ruby-lsp if available\n\
  if bundle show ruby-lsp > /dev/null 2>&1; then\n\
    echo "Using bundled ruby-lsp"\n\
    bundle exec ruby-lsp --stdio\n\
  else\n\
    echo "Using global ruby-lsp"\n\
    ruby-lsp --stdio\n\
  fi\n\
else\n\
  ruby-lsp --stdio\n\
fi' > /usr/local/bin/ruby-lsp-wrapper

RUN chmod +x /usr/local/bin/ruby-lsp-wrapper

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

# rust-analyzer
RUN curl -LO "https://github.com/rust-lang/rust-analyzer/releases/download/2025-01-20/rust-analyzer-x86_64-unknown-linux-gnu.gz" \
    && gzip -cd rust-analyzer-x86_64-unknown-linux-gnu.gz > /bin/rust-analyzer \
    && chmod +x /bin/rust-analyzer