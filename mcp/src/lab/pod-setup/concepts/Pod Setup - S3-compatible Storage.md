---
description: "Object storage in a pod for an app on an S3 SDK — AWS S3, R2, Spaces, MinIO: a RustFS container with a bucket made for the app."
parent: Pod Setup - Local Services
---
Object storage in a pod for an app on an S3 SDK — AWS S3, R2, Spaces, MinIO: a RustFS container with a bucket made for the app.

- A `rustfs` compose service (`image: rustfs/rustfs:1.0.1`), `ports:
  ["9000:9000", "9001:9001"]` (the S3 API and the console),
  `RUSTFS_ACCESS_KEY: rustfsadmin`, `RUSTFS_SECRET_KEY: rustfsadmin`, a
  named volume for `/data`, and a healthcheck: `test: ["CMD", "curl",
  "-fsS", "http://127.0.0.1:9000/health"]`. Never MinIO: its images are
  gone from Docker Hub and Quay, so `minio/minio` and `minio/mc` do not
  pull.
- A `rustfs-init` compose service (`image: amazon/aws-cli`) that waits on
  the healthcheck (`depends_on: { rustfs: { condition: service_healthy }
  }`) and makes the bucket, with `AWS_ACCESS_KEY_ID: rustfsadmin`,
  `AWS_SECRET_ACCESS_KEY: rustfsadmin`, `AWS_REGION: us-east-1` and
  `entrypoint: ["sh", "-c", "aws --endpoint-url http://rustfs:9000 s3api
  head-bucket --bucket <bucket> || aws --endpoint-url http://rustfs:9000
  s3 mb s3://<bucket>"]` — inside the compose file a service's name is its
  host, and a second boot finds the bucket instead of failing. The bucket
  is the name the repository's env uses, else `dev-bucket`; it is made
  there, not in the app's pre-start.
- The app's endpoint variables (`S3_ENDPOINT`, `AWS_ENDPOINT_URL`,
  `R2_ENDPOINT_URL`, `STORAGE_ENDPOINT`, …) are `http://127.0.0.1:9000` —
  an IP, so SDKs that put the bucket in the host name do not ask for
  `bucket.localhost`. The access key variables are `rustfsadmin`, the
  secret key variables `rustfsadmin`, and the region variables
  (`AWS_REGION`, `S3_REGION`, …) `us-east-1`: an AWS SDK will not make a
  client without a region.
- An app on Google Cloud Storage's own SDK (`@google-cloud/storage`,
  `google-cloud-storage`) does not speak S3: it gets a `fake-gcs-server`
  compose service instead (`image: fsouza/fake-gcs-server`, `command:
  -scheme http -port 4443`, `ports: ["4443:4443"]`) and
  `STORAGE_EMULATOR_HOST` `http://127.0.0.1:4443`.
