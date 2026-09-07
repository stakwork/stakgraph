// S3 log-payload offload
// ----------------------
// When BIFROST_S3_BUCKET is set at boot, the wrapper injects a
// logs_store.object_storage block into the materialised config.json
// so Bifrost offloads LLM request/response bodies to S3 while keeping
// searchable metadata (including signature-bound identity dims) in
// the local SQLite logs.db. Absent the bucket env, the written file
// is byte-identical to the seed — today's local-only behaviour.
//
// Credentials are ALWAYS emitted as Bifrost `env.<NAME>` references
// (never as the resolved secret). The wrapper never writes plaintext
// AWS keys onto the /app/data volume.
//
// Bucket-side requirements (operator, mandatory)
// ----------------------------------------------
// Full LLM payloads bound to identity dims are sensitive. The target
// bucket MUST have:
//
//   - SSE-KMS encryption
//   - S3 Block Public Access enabled
//   - a deny-non-TLS bucket policy
//   - S3 Object Lock (compliance mode) OR a deny-DeleteObject policy
//     so traces cannot be deleted by the injected credential
//   - the IAM user/role scoped to least privilege: s3:PutObject and
//     s3:GetObject on this bucket+prefix only (no s3:DeleteObject)
//
// Retention is governed by a single knob: client.log_retention_days
// in the seed (36500 ≈ 100 years). Do not add a competing
// logs_store.retention_days — Bifrost's cleaner treats values < 1 as
// "use the 365-day default", which would silently re-enable purge.
// HybridLogStore.DeleteLogsBatch does not delete S3 objects (it
// expects a bucket lifecycle); Object Lock / deny-DeleteObject is
// what actually keeps payloads around.
//
// Prefix layout
// -------------
// Default prefix is "bifrost". Operators should set BIFROST_S3_PREFIX
// to include an org/realm id (e.g. "bifrost/org_acme") so a future
// per-org authorization fix does not require a data migration.
// Bifrost's object key is `{prefix}/logs/YYYY/MM/DD/HH/{id}.json.gz`.
//
// Known limitation (pre-existing, out of scope)
// ---------------------------------------------
// `/_plugin/runs/` and `/_plugin/users/` read payloads via a single
// shared admin credential to Bifrost's `/api/logs`, with no per-org
// ownership check. This feature does not change that authorization
// surface. The prefix layout above is the forward-looking seam.
//
// Compress is intentionally omitted: Bifrost's Get() only decompresses
// when the stored object has ContentEncoding=gzip, which is set only
// when compress=true was on at write time. Ship uncompressed until
// read-back rehydration is proven in production.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	envS3Bucket    = "BIFROST_S3_BUCKET"
	envS3Region    = "BIFROST_S3_REGION"
	envS3AccessKey = "BIFROST_S3_ACCESS_KEY_ID"
	envS3SecretKey = "BIFROST_S3_SECRET_ACCESS_KEY"
	envS3Prefix    = "BIFROST_S3_PREFIX"
	envS3Endpoint  = "BIFROST_S3_ENDPOINT"

	// envS3ForcePathStyle enables S3 path-style URLs (required for
	// MinIO / LocalStack). Recognised truthy values: 1, true, yes.
	envS3ForcePathStyle = "BIFROST_S3_FORCE_PATH_STYLE"

	// defaultS3Prefix matches Bifrost's own objectstore default and
	// the upstream withobjectstorages3 example. Override via
	// BIFROST_S3_PREFIX to namespace per org/realm.
	defaultS3Prefix = "bifrost"

	// envRefAccessKey / envRefSecretKey are the literal strings we
	// write into config.json. Bifrost's SecretVar resolver expands
	// `env.NAME` at load time. Never substitute the real values here.
	envRefAccessKey = "env." + envS3AccessKey
	envRefSecretKey = "env." + envS3SecretKey
)

// s3OffloadInfo is the resolved, loggable view of S3 env. Credentials
// are never stored here — only whether they were present — so it is
// safe to print.
type s3OffloadInfo struct {
	Enabled        bool
	Bucket         string
	Region         string
	Prefix         string
	Endpoint       string
	ForcePathStyle bool
	HasAccessKey   bool
	HasSecretKey   bool
}

// logsStoreBlock is the typed logs_store object we marshal into
// config.json. Built with encoding/json, never string concatenation,
// so operator-supplied bucket/region/prefix cannot inject sibling
// keys or corrupt the document.
type logsStoreBlock struct {
	Enabled       bool               `json:"enabled"`
	Type          string             `json:"type"`
	Config        logsStoreSQLite    `json:"config"`
	ObjectStorage objectStorageBlock `json:"object_storage"`
}

type logsStoreSQLite struct {
	Path string `json:"path"`
}

type objectStorageBlock struct {
	Type            string `json:"type"`
	Bucket          string `json:"bucket"`
	Region          string `json:"region,omitempty"`
	Prefix          string `json:"prefix,omitempty"`
	AccessKeyID     string `json:"access_key_id,omitempty"`
	SecretAccessKey string `json:"secret_access_key,omitempty"`
	Endpoint        string `json:"endpoint,omitempty"`
	ForcePathStyle  bool   `json:"force_path_style,omitempty"`
}

// S3 bucket names: 3–63 chars, lowercase alphanumeric + dots/hyphens,
// must start and end alphanumeric. Hostile JSON fragments fail this.
var bucketNameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$`)

// AWS region ids (us-east-1, eu-central-1, us-gov-west-1, cn-north-1).
var regionNameRe = regexp.MustCompile(`^[a-z]{2}(-[a-z0-9]+)+-\d+$`)

// S3 key prefix: starts alphanumeric, then alnum / _ . - /. No JSON
// metacharacters, no spaces, no `..` path segments (checked separately).
var prefixRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9/_.-]*$`)

// Custom S3 endpoint (MinIO / LocalStack / R2). Scheme-optional host
// plus optional port and path. Quotes / braces / spaces rejected.
var endpointRe = regexp.MustCompile(`^(https?://)?[A-Za-z0-9._-]+(:[0-9]{1,5})?(/[A-Za-z0-9._/-]*)?$`)

func readS3OffloadEnv() s3OffloadInfo {
	info := s3OffloadInfo{
		Bucket:         strings.TrimSpace(os.Getenv(envS3Bucket)),
		Region:         strings.TrimSpace(os.Getenv(envS3Region)),
		Prefix:         strings.TrimSpace(os.Getenv(envS3Prefix)),
		Endpoint:       strings.TrimSpace(os.Getenv(envS3Endpoint)),
		HasAccessKey:   strings.TrimSpace(os.Getenv(envS3AccessKey)) != "",
		HasSecretKey:   strings.TrimSpace(os.Getenv(envS3SecretKey)) != "",
		ForcePathStyle: isTruthy(os.Getenv(envS3ForcePathStyle)),
	}
	if info.Prefix == "" {
		info.Prefix = defaultS3Prefix
	} else {
		info.Prefix = strings.Trim(info.Prefix, "/")
		if info.Prefix == "" {
			info.Prefix = defaultS3Prefix
		}
	}
	info.Enabled = info.Bucket != ""
	return info
}

func isTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes":
		return true
	default:
		return false
	}
}

// materializeConfig returns the bytes that should land at
// appDir/config.json: the seed as-is when S3 is off, or the seed
// plus a marshalled logs_store.object_storage block when S3 is on.
func materializeConfig(seed []byte, appDir string) ([]byte, s3OffloadInfo, error) {
	return materializeConfigFrom(seed, appDir, readS3OffloadEnv())
}

func materializeConfigFrom(seed []byte, appDir string, info s3OffloadInfo) ([]byte, s3OffloadInfo, error) {
	if !info.Enabled {
		return seed, info, nil
	}
	if err := validateS3Offload(info); err != nil {
		return nil, info, err
	}

	// Unmarshal into RawMessage values so every seed section other
	// than logs_store is preserved byte-for-byte. A map[string]any
	// round-trip would scramble nested key order and break the
	// desired-vs-on-disk idempotency compare.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(seed, &raw); err != nil {
		return nil, info, fmt.Errorf("parse seed config: %w", err)
	}

	block, err := json.Marshal(buildLogsStore(info, appDir))
	if err != nil {
		return nil, info, fmt.Errorf("marshal logs_store: %w", err)
	}
	raw["logs_store"] = block

	out, err := encodeTopLevel(raw)
	if err != nil {
		return nil, info, err
	}
	return out, info, nil
}

func buildLogsStore(info s3OffloadInfo, appDir string) logsStoreBlock {
	obj := objectStorageBlock{
		Type:   "s3",
		Bucket: info.Bucket,
		Region: info.Region,
		Prefix: info.Prefix,
	}
	// Static keys → env.NAME references, never the resolved secret.
	// Both omitted → Bifrost uses the default AWS credential chain
	// (instance role, IRSA, env AWS_ACCESS_KEY_ID, etc.).
	// Only emit the pair when BOTH env vars are present: Bifrost
	// rejects a half-configured static-credential block at boot.
	if info.HasAccessKey && info.HasSecretKey {
		obj.AccessKeyID = envRefAccessKey
		obj.SecretAccessKey = envRefSecretKey
	}
	if info.Endpoint != "" {
		obj.Endpoint = info.Endpoint
	}
	if info.ForcePathStyle {
		obj.ForcePathStyle = true
	}
	return logsStoreBlock{
		Enabled: true,
		Type:    "sqlite",
		Config: logsStoreSQLite{
			Path: filepath.Join(appDir, "logs.db"),
		},
		ObjectStorage: obj,
	}
}

// encodeTopLevel writes a JSON object with sorted keys so the output
// is deterministic across boots (Go map iteration is randomised).
// Nested values stay as the original RawMessage bytes.
func encodeTopLevel(raw map[string]json.RawMessage) ([]byte, error) {
	keys := make([]string, 0, len(raw))
	for k := range raw {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var buf bytes.Buffer
	buf.WriteString("{\n")
	for i, k := range keys {
		keyJSON, err := json.Marshal(k)
		if err != nil {
			return nil, fmt.Errorf("marshal key %q: %w", k, err)
		}
		val, err := indentRaw(raw[k])
		if err != nil {
			return nil, fmt.Errorf("encode %s: %w", k, err)
		}
		buf.WriteString("  ")
		buf.Write(keyJSON)
		buf.WriteString(": ")
		buf.Write(val)
		if i < len(keys)-1 {
			buf.WriteByte(',')
		}
		buf.WriteByte('\n')
	}
	buf.WriteString("}\n")
	return buf.Bytes(), nil
}

// indentRaw pretty-prints a RawMessage with 2-space indent so a
// compact logs_store blob (from json.Marshal) sits at the same
// indentation as the rest of the seed.
func indentRaw(raw json.RawMessage) ([]byte, error) {
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "  ", "  "); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func validateS3Offload(info s3OffloadInfo) error {
	if err := validateBucket(info.Bucket); err != nil {
		return err
	}
	if info.Region != "" {
		if err := validateRegion(info.Region); err != nil {
			return err
		}
	}
	if err := validatePrefix(info.Prefix); err != nil {
		return err
	}
	if info.Endpoint != "" {
		if err := validateEndpoint(info.Endpoint); err != nil {
			return err
		}
	}
	return nil
}

func validateBucket(v string) error {
	if !bucketNameRe.MatchString(v) || strings.Contains(v, "..") {
		return fmt.Errorf("invalid %s %q: must be a 3-63 char S3 bucket name (lowercase letters, digits, dots, hyphens)", envS3Bucket, v)
	}
	return nil
}

func validateRegion(v string) error {
	if !regionNameRe.MatchString(v) {
		return fmt.Errorf("invalid %s %q: must be an AWS region id (e.g. us-east-1)", envS3Region, v)
	}
	return nil
}

func validatePrefix(v string) error {
	if !prefixRe.MatchString(v) || strings.Contains(v, "..") {
		return fmt.Errorf("invalid %s %q: must be an S3 key prefix (alphanumeric, '/', '_', '.', '-')", envS3Prefix, v)
	}
	return nil
}

func validateEndpoint(v string) error {
	if !endpointRe.MatchString(v) {
		return fmt.Errorf("invalid %s %q: must be a host[:port] or http(s) URL", envS3Endpoint, v)
	}
	return nil
}

func logS3Offload(logger *log.Logger, info s3OffloadInfo) {
	if !info.Enabled {
		logger.Printf("S3 log offload disabled")
		return
	}
	logger.Printf("S3 log offload enabled bucket=%s region=%s prefix=%s",
		info.Bucket, info.Region, info.Prefix)
	if info.Region == "" {
		logger.Printf("WARNING: %s is set but %s is empty; Bifrost will use the AWS SDK default region chain",
			envS3Bucket, envS3Region)
	}
	switch {
	case info.HasAccessKey && info.HasSecretKey:
		// Static keys present as env refs — nothing to warn about.
	case !info.HasAccessKey && !info.HasSecretKey:
		logger.Printf("WARNING: %s is set but %s/%s are empty; Bifrost will use the default AWS credential chain (instance role)",
			envS3Bucket, envS3AccessKey, envS3SecretKey)
	default:
		logger.Printf("WARNING: %s is set but only one of %s/%s is present; both are required for static credentials",
			envS3Bucket, envS3AccessKey, envS3SecretKey)
	}
}

// writeFileAtomic writes data to path via a sibling temp file + rename
// so a torn write cannot leave Bifrost parsing half a JSON document.
func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmpPath := path + ".tmp"
	dst, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("create temp %s: %w", tmpPath, err)
	}
	if _, err := dst.Write(data); err != nil {
		_ = dst.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("write temp: %w", err)
	}
	if err := dst.Sync(); err != nil {
		_ = dst.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("fsync temp: %w", err)
	}
	if err := dst.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename temp -> %s: %w", path, err)
	}
	return nil
}
