# Phase 2 — Publish the gateway image via CI

> Build the `stakgraph-gateway` Docker image on every release and push
> it to GHCR. This is the prerequisite for phase-3 (swarm handoff) —
> sphinx-swarm needs a pull-able image tag to point at, not a
> `make docker-build`-on-laptop artifact.

## What lands

A new workflow at `.github/workflows/publish-gateway.yml` that:

1. Fires on `release: published` (same trigger as
   `publish-standalone.yml` — see `.github/workflows/publish-standalone.yml`
   for the canonical pattern).
2. Builds `gateway/Dockerfile` with `context: gateway/`.
3. Pushes to GHCR as both `:${{ github.ref_name }}` and `:latest`.
4. Caches the build via GHCR's `:buildcache` tag — the gateway build
   is heavy (clones bifrost, builds bifrost-http + Node UI + plugin
   + wrapper) so cache hits matter.

## Skeleton

Copy-and-modify of `publish-standalone.yml`. The diffs are:

- `name:` → `Build and Publish Gateway Docker Image`
- `context:` → `gateway` (not `.`)
- `file:` → `gateway/Dockerfile`
- Image name → `ghcr.io/stakwork/stakgraph-gateway`
- Build cache ref → `ghcr.io/stakwork/stakgraph-gateway:buildcache`

```yaml
name: Build and Publish Gateway Docker Image

on:
  release:
    types: [published]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: gateway
          file: gateway/Dockerfile
          platforms: linux/amd64
          push: true
          cache-from: type=registry,ref=ghcr.io/stakwork/stakgraph-gateway:buildcache
          cache-to: type=registry,ref=ghcr.io/stakwork/stakgraph-gateway:buildcache,mode=max
          provenance: false
          sbom: false
          tags: |
            ghcr.io/stakwork/stakgraph-gateway:${{ github.ref_name }}
            ghcr.io/stakwork/stakgraph-gateway:latest
```

## Notes / gotchas

- **Architecture.** `linux/amd64` only, matching `publish-standalone`.
  Multi-arch (`linux/arm64`) is a follow-up — the bifrost-http build
  inside the image uses CGO + Go plugins, which are arch-specific, so
  going multi-arch means either QEMU emulation (slow) or self-hosted
  arm runners. Defer until someone actually wants arm.
- **Build time.** Expect 8–15 min cold (UI npm install + bifrost
  clone + two Go builds). Warm cache should be 2–4 min. The 90-min
  `timeout-minutes` from `publish-standalone` is plenty of headroom.
- **Image size.** Final image is ~120 MB (alpine + dynamic libc +
  bifrost-http + plugin .so + wrapper). Stays well under any
  registry size limits.
- **`BIFROST_VERSION` pinning.** The Dockerfile has
  `ARG BIFROST_VERSION=transports/v1.5.2`. We don't override it in
  the workflow — releases of stakgraph pin the bifrost version they
  test against. To bump bifrost, edit the Dockerfile, retest locally,
  and cut a new stakgraph release.

## How to test before merging

```bash
# Build locally with the exact context the workflow uses
docker build -f gateway/Dockerfile -t test-gateway gateway

# Run smoke tests
cd gateway && make docker-up && curl -s http://localhost:8181/health
```

If `make docker-up` works, the workflow will work — the workflow is
just `docker build` with a different output destination.

## After this ships

Once an image exists at `ghcr.io/stakwork/stakgraph-gateway:<tag>`:

1. **Phase 3 unblocks.** sphinx-swarm's `bifrost.rs` can flip its
   image reference from `maximhq/bifrost` to
   `ghcr.io/stakwork/stakgraph-gateway` (or whatever org we publish
   under).
2. **Operators can pull directly.** Anyone running stakgraph outside
   the swarm context can `docker pull
   ghcr.io/stakwork/stakgraph-gateway:latest` and follow the gateway
   README's quick-start.
3. **Versioning is explicit.** Release tags become the image tags,
   so it's obvious which bifrost + plugin combination is in any
   given deployment.

## What this doesn't cover

- **Multi-arch builds.** See note above.
- **Image signing (cosign / sigstore).** Not done for
  `publish-standalone` either; defer to a follow-up that does both
  at once.
- **SBOM generation.** `sbom: false` matches `publish-standalone`.
  Worth turning on when we have a security-review checklist that
  consumes it, not before.
- **Auto-bumping bifrost version.** Manual, on purpose — plugin
  compatibility is fragile (`-buildmode=plugin` requires
  byte-identical deps with bifrost-http) and we want a human
  signing off on every bump.
