---
description: "Object storage in a pod — S3, R2, GCS-compatible: a MinIO container with a bucket made for the app."
parent: Pod Setup - Local Services
---
Object storage in a pod — S3, R2, GCS-compatible: a MinIO container with a bucket made for the app.

- A `minio` compose service (`image: minio/minio`, `command: server /data
  --console-address ":9001"`), `ports: ["9000:9000", "9001:9001"]`,
  `MINIO_ROOT_USER: minioadmin`, `MINIO_ROOT_PASSWORD: minioadmin`, a named
  volume for `/data`, and a healthcheck.
- A `minio-init` compose service (`image: minio/mc`) that depends on the
  healthcheck and makes the bucket: the bucket name the repository's env
  uses, else `dev-bucket`. The bucket is made there, not in the app's
  pre-start.
- The app's endpoint variables (`S3_ENDPOINT`, `AWS_ENDPOINT_URL`,
  `R2_ENDPOINT_URL`, `STORAGE_ENDPOINT`, …) are `http://127.0.0.1:9000` —
  `127.0.0.1`, not `localhost`, so SDKs that put the bucket in the host
  name do not resolve `bucket.localhost`. The access key variables are
  `minioadmin`, the secret key variables `minioadmin`.
