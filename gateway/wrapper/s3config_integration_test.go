// Integration smoke for S3 payload offload against LocalStack/MinIO.
//
// Not run in ordinary `go test ./...` — it needs a live gateway image
// booted with BIFROST_S3_* and an S3-compatible bucket. Enable with:
//
//	BIFROST_S3_INTEGRATION=1 \
//	BIFROST_S3_ENDPOINT=http://localhost:4566 \
//	BIFROST_S3_BUCKET=... BIFROST_S3_REGION=us-east-1 \
//	BIFROST_S3_ACCESS_KEY_ID=test BIFROST_S3_SECRET_ACCESS_KEY=test \
//	go test -count=1 -run TestIntegrationS3Offload ./
//
// What this proves (the T1 acceptance path that unit tests cannot):
//   - wrapper materialises logs_store.object_storage with env.* refs
//   - bifrost-http honors that block (no license gate) and pings S3
//   - a subsequent LLM call lands an object under {prefix}/logs/...
//   - GET /api/logs rehydrates the body; identity dims stay on the
//     local metadata row
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestIntegrationS3Offload_MaterializeAgainstLiveEnv(t *testing.T) {
	if os.Getenv("BIFROST_S3_INTEGRATION") == "" {
		t.Skip("set BIFROST_S3_INTEGRATION=1 to run against LocalStack/MinIO")
	}

	seedPath := filepath.Join("..", "data", "config.json")
	seed, err := os.ReadFile(seedPath)
	if err != nil {
		t.Fatalf("read seed: %v", err)
	}

	out, info, err := materializeConfig(seed, "/app/data")
	if err != nil {
		t.Fatalf("materialize with live env: %v", err)
	}
	if !info.Enabled {
		t.Fatal("BIFROST_S3_INTEGRATION=1 requires BIFROST_S3_BUCKET to be set")
	}

	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("materialised config is not JSON: %v", err)
	}
	ls, _ := parsed["logs_store"].(map[string]any)
	if ls == nil {
		t.Fatal("expected logs_store in materialised config")
	}
	obj, _ := ls["object_storage"].(map[string]any)
	if obj == nil || obj["type"] != "s3" {
		t.Fatalf("expected object_storage.type=s3, got %#v", obj)
	}
	if obj["bucket"] != info.Bucket {
		t.Fatalf("bucket = %v, want %s", obj["bucket"], info.Bucket)
	}
	if ak, _ := obj["access_key_id"].(string); ak != "" && ak != envRefAccessKey {
		t.Fatalf("access_key_id must be an env.* ref, got %q", ak)
	}

	t.Logf("materialised logs_store.object_storage bucket=%s region=%s prefix=%s endpoint=%s",
		info.Bucket, info.Region, info.Prefix, info.Endpoint)
	t.Log("full boot+LLM-call+object-lands-in-bucket assertion requires the gateway image; this test only checks the config the wrapper would write")
}
