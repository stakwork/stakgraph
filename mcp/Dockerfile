# docker buildx build --platform linux/amd64,linux/arm64 -t repo2graph .

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

# Install dependencies in src/aieo
RUN cd src/aieo && yarn

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

EXPOSE 3000

WORKDIR /usr/src/app

# Define the command to run your app
CMD ["node", "build/index.js"]
