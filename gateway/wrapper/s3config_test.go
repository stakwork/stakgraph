// Tests for conditional logs_store.object_storage injection.
//
// These pin the T1 contract:
//   (a) bucket set  → logs_store.object_storage emitted with env.<NAME>
//       credential refs, no plaintext key, log_retention_days == 36500
//   (b) bucket unset → output byte-identical to the seed
//   (c) hostile bucket/prefix is rejected, not written raw
//   (d) second boot with unchanged env is a no-op (no mtime bump)
//
// No Bifrost, no Docker, no network. Run with:
//
//	go test ./...
package main

import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const testSeed = `{
  "$schema": "https://www.getbifrost.ai/schema",
  "client": {
    "log_retention_days": 36500,
    "drop_excess_requests": false,
    "enforce_auth_on_inference": true
  },
  "auth_config": {
    "admin_username": "env.BIFROST_ADMIN_USER",
    "admin_password": "env.BIFROST_ADMIN_PASS"
  },
  "config_store": {
    "enabled": true,
    "type": "sqlite",
    "config": {
      "path": "/app/data/config.db"
    }
  }
}
`

const (
	testBucket      = "stakgraph-llm-logs"
	testRegion      = "us-east-1"
	testPrefix      = "bifrost/org_acme"
	testPlainKey    = "AKIAIOSFODNN7EXAMPLE"
	testPlainSecret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
)

func TestMaterializeConfig_BucketUnset_ByteIdenticalToSeed(t *testing.T) {
	out, info, err := materializeConfigFrom([]byte(testSeed), "/app/data", s3OffloadInfo{})
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	if info.Enabled {
		t.Fatal("expected S3 offload disabled")
	}
	if !bytes.Equal(out, []byte(testSeed)) {
		t.Fatalf("output must be byte-identical to seed when bucket unset\ngot:\n%s", out)
	}
}

func TestMaterializeConfig_BucketSet_InjectsLogsStore(t *testing.T) {
	info := s3OffloadInfo{
		Enabled:      true,
		Bucket:       testBucket,
		Region:       testRegion,
		Prefix:       testPrefix,
		HasAccessKey: true,
		HasSecretKey: true,
	}
	out, got, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	if !got.Enabled {
		t.Fatal("expected S3 offload enabled")
	}

	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}

	client, _ := parsed["client"].(map[string]any)
	if client == nil {
		t.Fatal("missing client")
	}
	if days, _ := client["log_retention_days"].(float64); days != 36500 {
		t.Fatalf("log_retention_days = %v, want 36500", client["log_retention_days"])
	}

	ls, _ := parsed["logs_store"].(map[string]any)
	if ls == nil {
		t.Fatalf("missing logs_store:\n%s", out)
	}
	if ls["enabled"] != true {
		t.Errorf("logs_store.enabled = %v, want true", ls["enabled"])
	}
	if ls["type"] != "sqlite" {
		t.Errorf("logs_store.type = %v, want sqlite", ls["type"])
	}
	cfg, _ := ls["config"].(map[string]any)
	if cfg["path"] != "/app/data/logs.db" {
		t.Errorf("logs_store.config.path = %v, want /app/data/logs.db", cfg["path"])
	}
	if _, ok := ls["retention_days"]; ok {
		t.Fatal("logs_store must not carry a competing retention_days field")
	}

	obj, _ := ls["object_storage"].(map[string]any)
	if obj == nil {
		t.Fatal("missing object_storage")
	}
	if obj["type"] != "s3" {
		t.Errorf("object_storage.type = %v, want s3", obj["type"])
	}
	if obj["bucket"] != testBucket {
		t.Errorf("bucket = %v, want %s", obj["bucket"], testBucket)
	}
	if obj["region"] != testRegion {
		t.Errorf("region = %v, want %s", obj["region"], testRegion)
	}
	if obj["prefix"] != testPrefix {
		t.Errorf("prefix = %v, want %s", obj["prefix"], testPrefix)
	}
	if obj["access_key_id"] != envRefAccessKey {
		t.Errorf("access_key_id = %v, want %s", obj["access_key_id"], envRefAccessKey)
	}
	if obj["secret_access_key"] != envRefSecretKey {
		t.Errorf("secret_access_key = %v, want %s", obj["secret_access_key"], envRefSecretKey)
	}
	if _, ok := obj["compress"]; ok {
		t.Fatal("compress must not be set (ship uncompressed until read-back is proven)")
	}

	if bytes.Contains(out, []byte(testPlainKey)) || bytes.Contains(out, []byte(testPlainSecret)) {
		t.Fatal("plaintext AWS key material must never appear in the written config")
	}
	if !bytes.Contains(out, []byte(envRefAccessKey)) || !bytes.Contains(out, []byte(envRefSecretKey)) {
		t.Fatal("credentials must be emitted as env.<NAME> references")
	}
}

func TestMaterializeConfig_NoStaticCreds_OmitsKeyFields(t *testing.T) {
	info := s3OffloadInfo{
		Enabled: true,
		Bucket:  testBucket,
		Region:  testRegion,
		Prefix:  defaultS3Prefix,
	}
	out, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("json: %v", err)
	}
	obj := parsed["logs_store"].(map[string]any)["object_storage"].(map[string]any)
	if _, ok := obj["access_key_id"]; ok {
		t.Fatal("access_key_id must be omitted when static creds are absent (instance-role path)")
	}
	if _, ok := obj["secret_access_key"]; ok {
		t.Fatal("secret_access_key must be omitted when static creds are absent")
	}
}

func TestMaterializeConfig_HalfCreds_OmitsKeyFields(t *testing.T) {
	// Bifrost rejects a half-configured static-credential block at boot.
	info := s3OffloadInfo{
		Enabled:      true,
		Bucket:       testBucket,
		Region:       testRegion,
		Prefix:       defaultS3Prefix,
		HasAccessKey: true,
		HasSecretKey: false,
	}
	out, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	if bytes.Contains(out, []byte("access_key_id")) {
		t.Fatal("half-configured static creds must not emit access_key_id")
	}
}

func TestMaterializeConfig_DefaultPrefix(t *testing.T) {
	info := s3OffloadInfo{
		Enabled: true,
		Bucket:  testBucket,
		Region:  testRegion,
		Prefix:  defaultS3Prefix,
	}
	out, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	var parsed map[string]any
	_ = json.Unmarshal(out, &parsed)
	obj := parsed["logs_store"].(map[string]any)["object_storage"].(map[string]any)
	if obj["prefix"] != defaultS3Prefix {
		t.Fatalf("prefix = %v, want %s", obj["prefix"], defaultS3Prefix)
	}
}

func TestMaterializeConfig_EndpointAndPathStyle(t *testing.T) {
	info := s3OffloadInfo{
		Enabled:        true,
		Bucket:         testBucket,
		Region:         testRegion,
		Prefix:         defaultS3Prefix,
		Endpoint:       "http://localhost:4566",
		ForcePathStyle: true,
		HasAccessKey:   true,
		HasSecretKey:   true,
	}
	out, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	var parsed map[string]any
	_ = json.Unmarshal(out, &parsed)
	obj := parsed["logs_store"].(map[string]any)["object_storage"].(map[string]any)
	if obj["endpoint"] != "http://localhost:4566" {
		t.Errorf("endpoint = %v", obj["endpoint"])
	}
	if obj["force_path_style"] != true {
		t.Errorf("force_path_style = %v, want true", obj["force_path_style"])
	}
}

func TestMaterializeConfig_Deterministic(t *testing.T) {
	info := s3OffloadInfo{
		Enabled:      true,
		Bucket:       testBucket,
		Region:       testRegion,
		Prefix:       testPrefix,
		HasAccessKey: true,
		HasSecretKey: true,
	}
	a, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatal("materializeConfigFrom must be deterministic (idempotency depends on it)")
	}
}

func TestValidate_HostileBucketRejected(t *testing.T) {
	hostile := []string{
		`", "injected_key":`,
		`foo", "injected_key": "x`,
		`../../etc/passwd`,
		`Bucket With Spaces`,
		`UPPERCASE`,
		`ab`, // too short
		"",
	}
	for _, b := range hostile {
		info := s3OffloadInfo{Enabled: true, Bucket: b, Region: testRegion, Prefix: defaultS3Prefix}
		_, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
		if err == nil {
			t.Errorf("bucket %q: want validation error, got nil", b)
		}
	}
}

func TestValidate_HostilePrefixRejected(t *testing.T) {
	hostile := []string{
		`", "injected_key":`,
		`foo/../bar`,
		`has space`,
		`{evil}`,
	}
	for _, p := range hostile {
		info := s3OffloadInfo{Enabled: true, Bucket: testBucket, Region: testRegion, Prefix: p}
		_, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info)
		if err == nil {
			t.Errorf("prefix %q: want validation error, got nil", p)
		}
	}
}

func TestValidate_HostileRegionRejected(t *testing.T) {
	info := s3OffloadInfo{Enabled: true, Bucket: testBucket, Region: `us-east-1","x":1`, Prefix: defaultS3Prefix}
	if _, _, err := materializeConfigFrom([]byte(testSeed), "/app/data", info); err == nil {
		t.Fatal("hostile region must be rejected")
	}
}

func TestJSONMarshalEscapesEvenIfValidationBypassed(t *testing.T) {
	// Defense in depth: encoding/json must quote a crafted bucket so it
	// cannot close the string and inject a sibling key.
	block := objectStorageBlock{
		Type:   "s3",
		Bucket: `foo", "injected_key": "pwned`,
		Region: testRegion,
		Prefix: defaultS3Prefix,
	}
	raw, err := json.Marshal(block)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(`"injected_key"`)) && !bytes.Contains(raw, []byte(`\"injected_key\"`)) && !bytes.Contains(raw, []byte(`\u0022injected_key`)) {
		// A raw (unescaped) injected_key key would mean concatenation won.
		var parsed map[string]any
		if err := json.Unmarshal(raw, &parsed); err != nil {
			t.Fatalf("marshal produced invalid JSON: %v", err)
		}
		if _, ok := parsed["injected_key"]; ok {
			t.Fatalf("hostile bucket injected a sibling key: %s", raw)
		}
	}
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("marshal produced invalid JSON: %v", err)
	}
	if _, ok := parsed["injected_key"]; ok {
		t.Fatalf("hostile bucket injected a sibling key: %s", raw)
	}
	if parsed["bucket"] != block.Bucket {
		t.Fatalf("bucket round-trip = %v", parsed["bucket"])
	}
}

func TestReadS3OffloadEnv(t *testing.T) {
	t.Setenv(envS3Bucket, "")
	t.Setenv(envS3Region, "")
	t.Setenv(envS3Prefix, "")
	t.Setenv(envS3AccessKey, "")
	t.Setenv(envS3SecretKey, "")
	t.Setenv(envS3Endpoint, "")
	t.Setenv(envS3ForcePathStyle, "")
	info := readS3OffloadEnv()
	if info.Enabled {
		t.Fatal("empty bucket must disable offload")
	}

	t.Setenv(envS3Bucket, "  "+testBucket+"  ")
	t.Setenv(envS3Region, testRegion)
	t.Setenv(envS3Prefix, testPrefix)
	t.Setenv(envS3AccessKey, testPlainKey)
	t.Setenv(envS3SecretKey, testPlainSecret)
	info = readS3OffloadEnv()
	if !info.Enabled || info.Bucket != testBucket || info.Region != testRegion || info.Prefix != testPrefix {
		t.Fatalf("unexpected info: %+v", info)
	}
	if !info.HasAccessKey || !info.HasSecretKey {
		t.Fatal("static creds should be detected")
	}
	if info.HasAccessKey && strings.Contains(info.Bucket, testPlainKey) {
		t.Fatal("info must not carry key material on Bucket")
	}
}

func TestMaterializeConfig_RealSeed(t *testing.T) {
	seedPath := filepath.Join("..", "data", "config.json")
	seed, err := os.ReadFile(seedPath)
	if err != nil {
		t.Skipf("seed not available: %v", err)
	}

	off, _, err := materializeConfigFrom(seed, "/app/data", s3OffloadInfo{})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(off, seed) {
		t.Fatal("real seed must pass through byte-identical when S3 is off")
	}

	on, _, err := materializeConfigFrom(seed, "/app/data", s3OffloadInfo{
		Enabled:      true,
		Bucket:       testBucket,
		Region:       testRegion,
		Prefix:       testPrefix,
		HasAccessKey: true,
		HasSecretKey: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(on, &parsed); err != nil {
		t.Fatalf("injected real seed is not JSON: %v", err)
	}
	if parsed["plugins"] == nil || parsed["providers"] == nil || parsed["auth_config"] == nil {
		t.Fatal("injection must preserve plugins/providers/auth_config")
	}
	ls := parsed["logs_store"].(map[string]any)
	obj := ls["object_storage"].(map[string]any)
	if obj["access_key_id"] != envRefAccessKey {
		t.Fatal("real-seed injection must use env.* credential refs")
	}
	if bytes.Contains(on, []byte(testPlainKey)) {
		t.Fatal("plaintext key in real-seed output")
	}
}

func TestSeedFile_LogRetentionDays(t *testing.T) {
	seedPath := filepath.Join("..", "data", "config.json")
	raw, err := os.ReadFile(seedPath)
	if err != nil {
		t.Skipf("seed not available: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("seed is not valid JSON: %v", err)
	}
	client, _ := parsed["client"].(map[string]any)
	if days, _ := client["log_retention_days"].(float64); days != 36500 {
		t.Fatalf("data/config.json client.log_retention_days = %v, want 36500", client["log_retention_days"])
	}
	if _, ok := parsed["logs_store"]; ok {
		t.Fatal("seed must not contain a logs_store block (injection is opt-in at boot)")
	}
}

func TestSyncSeedConfig_S3Off_ByteIdenticalAndIdempotent(t *testing.T) {
	t.Setenv(envS3Bucket, "")
	t.Setenv(envS3Region, "")
	t.Setenv(envS3Prefix, "")
	t.Setenv(envS3AccessKey, "")
	t.Setenv(envS3SecretKey, "")

	dir := t.TempDir()
	seedPath := filepath.Join(dir, "config.json.seed")
	appDir := filepath.Join(dir, "data")
	if err := os.WriteFile(seedPath, []byte(testSeed), 0o644); err != nil {
		t.Fatal(err)
	}
	logger := log.New(os.Stderr, "[test] ", 0)
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("first sync: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(appDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, []byte(testSeed)) {
		t.Fatalf("S3-off output must equal seed\ngot:\n%s", got)
	}

	dst := filepath.Join(appDir, "config.json")
	past := time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC)
	if err := os.Chtimes(dst, past, past); err != nil {
		t.Fatal(err)
	}
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("second sync: %v", err)
	}
	st, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !st.ModTime().Equal(past) {
		t.Fatalf("idempotent S3-off sync bumped mtime: got %v want %v", st.ModTime(), past)
	}
}

func TestSyncSeedConfig_S3On_EnvRefsNoPlaintextIdempotent(t *testing.T) {
	t.Setenv(envS3Bucket, testBucket)
	t.Setenv(envS3Region, testRegion)
	t.Setenv(envS3Prefix, testPrefix)
	t.Setenv(envS3AccessKey, testPlainKey)
	t.Setenv(envS3SecretKey, testPlainSecret)

	dir := t.TempDir()
	seedPath := filepath.Join(dir, "config.json.seed")
	appDir := filepath.Join(dir, "data")
	if err := os.WriteFile(seedPath, []byte(testSeed), 0o644); err != nil {
		t.Fatal(err)
	}
	logger := log.New(os.Stderr, "[test] ", 0)
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("first sync: %v", err)
	}

	dst := filepath.Join(appDir, "config.json")
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(got, []byte(testSeed)) {
		t.Fatal("S3-on output must differ from seed (logs_store injected)")
	}
	if bytes.Contains(got, []byte(testPlainKey)) || bytes.Contains(got, []byte(testPlainSecret)) {
		t.Fatal("plaintext key material leaked into config.json")
	}
	if !bytes.Contains(got, []byte(envRefAccessKey)) || !bytes.Contains(got, []byte(envRefSecretKey)) {
		t.Fatal("missing env.<NAME> credential references")
	}
	if !bytes.Contains(got, []byte(`"bucket": "`+testBucket+`"`)) {
		t.Fatalf("missing bucket in output:\n%s", got)
	}

	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("written config is not JSON: %v", err)
	}
	client := parsed["client"].(map[string]any)
	if days, _ := client["log_retention_days"].(float64); days != 36500 {
		t.Fatalf("log_retention_days = %v, want 36500", client["log_retention_days"])
	}

	past := time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC)
	if err := os.Chtimes(dst, past, past); err != nil {
		t.Fatal(err)
	}
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("second sync: %v", err)
	}
	st, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !st.ModTime().Equal(past) {
		t.Fatalf("idempotent S3-on sync bumped mtime: got %v want %v", st.ModTime(), past)
	}

	// Changing env must rewrite.
	t.Setenv(envS3Prefix, "bifrost/other-org")
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("third sync: %v", err)
	}
	st, err = os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if st.ModTime().Equal(past) {
		t.Fatal("changing BIFROST_S3_PREFIX should rewrite config.json")
	}
	rewritten, _ := os.ReadFile(dst)
	if !bytes.Contains(rewritten, []byte("bifrost/other-org")) {
		t.Fatalf("rewritten config missing new prefix:\n%s", rewritten)
	}
}

func TestSyncSeedConfig_HostileBucketFallsBackToSeed(t *testing.T) {
	t.Setenv(envS3Bucket, `foo", "injected_key": "pwned`)
	t.Setenv(envS3Region, testRegion)
	t.Setenv(envS3Prefix, defaultS3Prefix)
	t.Setenv(envS3AccessKey, testPlainKey)
	t.Setenv(envS3SecretKey, testPlainSecret)

	dir := t.TempDir()
	seedPath := filepath.Join(dir, "config.json.seed")
	appDir := filepath.Join(dir, "data")
	if err := os.WriteFile(seedPath, []byte(testSeed), 0o644); err != nil {
		t.Fatal(err)
	}
	logger := log.New(os.Stderr, "[test] ", 0)
	if err := syncSeedConfig(logger, seedPath, appDir); err != nil {
		t.Fatalf("sync: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(appDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, []byte(testSeed)) {
		t.Fatalf("hostile bucket must fall back to seed, not write injected JSON\ngot:\n%s", got)
	}
	if bytes.Contains(got, []byte("injected_key")) {
		t.Fatal("injected_key leaked into written config")
	}
}

func TestLogS3Offload_NeverPrintsSecrets(t *testing.T) {
	var buf bytes.Buffer
	logger := log.New(&buf, "", 0)
	logS3Offload(logger, s3OffloadInfo{
		Enabled:      true,
		Bucket:       testBucket,
		Region:       testRegion,
		Prefix:       testPrefix,
		HasAccessKey: true,
		HasSecretKey: true,
	})
	s := buf.String()
	if !strings.Contains(s, testBucket) || !strings.Contains(s, testRegion) {
		t.Fatalf("expected bucket+region in log, got %q", s)
	}
	if strings.Contains(s, testPlainKey) || strings.Contains(s, testPlainSecret) {
		t.Fatalf("secrets leaked in log: %q", s)
	}

	buf.Reset()
	logS3Offload(logger, s3OffloadInfo{Enabled: true, Bucket: testBucket})
	s = buf.String()
	if !strings.Contains(s, "WARNING") {
		t.Fatalf("missing WARNING for bucket-without-region/creds: %q", s)
	}
}
