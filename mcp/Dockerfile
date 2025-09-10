# docker buildx build --platform linux/amd64,linux/arm64 -t repo2graph .

# Use an official Node.js runtime as a parent image
FROM node:22-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies and build tools
RUN apt update && apt install -y sed git vim python3 make g++ && \
    npx -y playwright install-deps && \
    npx -y playwright install && \
    rm -rf /var/lib/apt/lists/*

# Copy package.json and yarn.lock
COPY package.json yarn.lock ./

# Copy workspace packages first (needed for workspace resolution)
COPY packages ./packages

# Install dependencies with workspace support
RUN yarn install --frozen-lockfile

# Rebuild native dependencies for current architecture
RUN npm rebuild --arch=$(uname -m)

# Copy the rest of your application code
COPY tsconfig.json tsconfig.build.json ./
COPY src ./src

# Copy additional files needed for runtime
COPY docs/redoc-static.html ./docs/redoc-static.html
COPY textarea ./textarea

# Build the application
RUN yarn build

EXPOSE 3000

# Define the command to run your app
CMD ["yarn", "start"]
