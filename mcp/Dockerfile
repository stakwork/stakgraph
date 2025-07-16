# docker build -t repo2graph .

# Use an official Node.js runtime as a parent image
FROM node:22-bookworm-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies
RUN apt update
RUN apt install -y sed git vim

RUN npx -y playwright install-deps
RUN npx -y playwright install

# Copy package.json and package-lock.json (if available)
COPY ./package.json ./package.json

# Install dependencies
RUN yarn

# Copy the rest of your application code
COPY ./tsconfig.json ./tsconfig.json
COPY ./src ./src
RUN mkdir -p ./docs
COPY ./docs/redoc-static.html ./docs/redoc-static.html
COPY ./textarea ./textarea
COPY ./app ./app
RUN mkdir -p ./tests
COPY ./tests/playwright.config.js ./tests/playwright.config.js

# Build the application
RUN yarn build

EXPOSE 3000

WORKDIR /usr/src/app

# Define the command to run your app
CMD ["node", "build/index.js"]
