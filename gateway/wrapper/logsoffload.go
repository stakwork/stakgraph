package main

// Log payload offload to S3-compatible object storage.
//
// Bifrost keeps every request's full payload (input history, output,
// params, tools, raw request/response, ...) in one SQLite file,
// <app-dir>/logs.db, and nothing ever shrinks it. Bifrost
// transports/v2.2.2 can instead run a "hybrid" log store: each log's
// payload is gzipped to <prefix>/logs/YYYY/MM/DD/HH/<id>.json.gz in a
// bucket, SQLite keeps a small index row (tokens, cost, metadata, a
// 2 KB summary, the last user message), and GET /api/logs/{id} fetches
// the payload back. It is switched on by a `logs_store.object_storage`
// block in config.json (framework/logstore/store.go NewLogStore).
//
// The wrapper builds that block from the env vars below, so a manager
// (sphinx-swarm, Stadeum) only sets env. The contract is fixed; the
// managers depend on it:
//
//	BIFROST_LOGS_S3_BUCKET             set (non-empty) turns offload on
//	BIFROST_LOGS_S3_PREFIX             key prefix, default "bifrost"
//	BIFROST_LOGS_S3_REGION             optional, passed through
//	BIFROST_LOGS_S3_ENDPOINT           optional, S3-compatible store (R2,
//	                                   MinIO); also turns on path-style
//	BIFROST_LOGS_S3_ACCESS_KEY_ID      optional, both or neither; neither
//	BIFROST_LOGS_S3_SECRET_ACCESS_KEY  means the AWS default credential
//	                                   chain (instance role, AWS_* env)
//
// With BIFROST_LOGS_S3_BUCKET unset or empty the written config.json is
// the seed, byte for byte, which is exactly the behaviour before this
// file existed.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

const (
	envLogsS3Bucket          = "BIFROST_LOGS_S3_BUCKET"
	envLogsS3Prefix          = "BIFROST_LOGS_S3_PREFIX"
	envLogsS3Region          = "BIFROST_LOGS_S3_REGION"
	envLogsS3Endpoint        = "BIFROST_LOGS_S3_ENDPOINT"
	envLogsS3AccessKeyID     = "BIFROST_LOGS_S3_ACCESS_KEY_ID"
	envLogsS3SecretAccessKey = "BIFROST_LOGS_S3_SECRET_ACCESS_KEY"

	// defaultLogsS3Prefix matches Bifrost's own default
	// (objectstore.Config.GetPrefix), so an unset prefix means the same
	// thing to both.
	defaultLogsS3Prefix = "bifrost"

	// logsStoreKey is the top-level config.json key Bifrost reads.
	logsStoreKey = "logs_store"
)

// logsOffload is the offload destination read from the env contract.
// A nil *logsOffload means offload is off.
type logsOffload struct {
	Bucket   string
	Prefix   string
	Region   string
	Endpoint string
	// StaticKeys is true when both key vars are set. The key values are
	// never copied into config.json: the block carries env.* references
	// that Bifrost resolves itself (schemas.SecretVar), so the secret is
	// never written to the volume.
	StaticKeys bool
}

// String describes the destination for log lines. No secrets.
func (o *logsOffload) String() string {
	if o == nil {
		return "off"
	}
	s := fmt.Sprintf("s3://%s/%s/logs/", o.Bucket, o.Prefix)
	if o.Endpoint != "" {
		s += " endpoint=" + o.Endpoint
	}
	if o.Region != "" {
		s += " region=" + o.Region
	}
	if o.StaticKeys {
		s += " credentials=static-keys"
	} else {
		s += " credentials=default-chain"
	}
	return s
}

// logsOffloadFromEnv reads the env contract. It returns (nil, nil) when
// offload is off (bucket unset or blank), and (nil, err) for a
// combination it refuses: exactly one of the two key vars set.
func logsOffloadFromEnv(getenv func(string) string) (*logsOffload, error) {
	bucket := strings.TrimSpace(getenv(envLogsS3Bucket))
	if bucket == "" {
		return nil, nil
	}

	hasID := strings.TrimSpace(getenv(envLogsS3AccessKeyID)) != ""
	hasSecret := strings.TrimSpace(getenv(envLogsS3SecretAccessKey)) != ""
	if hasID != hasSecret {
		set, unset := envLogsS3AccessKeyID, envLogsS3SecretAccessKey
		if hasSecret {
			set, unset = unset, set
		}
		return nil, fmt.Errorf("%s is set but %s is not: set both, or neither to use the default AWS credential chain",
			set, unset)
	}

	// "a/b/" and "/a/b" would give keys like "a/b//logs/...": Bifrost
	// joins prefix and "logs" with a slash of its own.
	prefix := strings.Trim(strings.TrimSpace(getenv(envLogsS3Prefix)), "/")
	if prefix == "" {
		prefix = defaultLogsS3Prefix
	}

	return &logsOffload{
		Bucket:     bucket,
		Prefix:     prefix,
		Region:     strings.TrimSpace(getenv(envLogsS3Region)),
		Endpoint:   strings.TrimSpace(getenv(envLogsS3Endpoint)),
		StaticKeys: hasID && hasSecret,
	}, nil
}

// logsStoreJSON is the `logs_store` block Bifrost's lib.ConfigData
// reads (framework/logstore/config.go, framework/objectstore/config.go).
// A struct, not a map, so field order (and so the written file) is
// stable across boots and the skip-when-identical check holds.
type logsStoreJSON struct {
	Enabled       bool              `json:"enabled"`
	Type          string            `json:"type"`
	Config        logsStoreSQLite   `json:"config"`
	ObjectStorage objectStorageJSON `json:"object_storage"`
}

type logsStoreSQLite struct {
	Path string `json:"path"`
}

type objectStorageJSON struct {
	Type            string `json:"type"`
	Bucket          string `json:"bucket"`
	Prefix          string `json:"prefix"`
	Compress        bool   `json:"compress"`
	Region          string `json:"region,omitempty"`
	Endpoint        string `json:"endpoint,omitempty"`
	ForcePathStyle  bool   `json:"force_path_style,omitempty"`
	AccessKeyID     string `json:"access_key_id,omitempty"`
	SecretAccessKey string `json:"secret_access_key,omitempty"`
}

// logsStoreBlock builds the block for o.
//
// The SQLite path must be the file Bifrost uses when config.json has no
// logs_store section: <app-dir>/logs.db (lib.LoadConfig's logsDBPath).
// An explicit, enabled logs_store is used as is, so any other path would
// open a fresh, empty logs.db and orphan every existing row.
func logsStoreBlock(o *logsOffload, appDir string) logsStoreJSON {
	store := objectStorageJSON{
		Type:     "s3",
		Bucket:   o.Bucket,
		Prefix:   o.Prefix,
		Compress: true,
		Region:   o.Region,
		Endpoint: o.Endpoint,
		// Path-style ("endpoint/bucket/key") is what MinIO and most
		// S3-compatible stores serve; virtual-hosted style needs a DNS
		// name per bucket. AWS itself keeps virtual-hosted (no endpoint).
		ForcePathStyle: o.Endpoint != "",
	}
	if o.StaticKeys {
		store.AccessKeyID = "env." + envLogsS3AccessKeyID
		store.SecretAccessKey = "env." + envLogsS3SecretAccessKey
	}
	return logsStoreJSON{
		Enabled:       true,
		Type:          "sqlite",
		Config:        logsStoreSQLite{Path: filepath.Join(appDir, "logs.db")},
		ObjectStorage: store,
	}
}

// errSeedHasLogsStore means the image's seed config.json already has a
// logs_store section. The seed then wins and the env vars are ignored,
// rather than the wrapper silently rewriting a block someone put there
// on purpose.
var errSeedHasLogsStore = errors.New("the seed config.json already has a logs_store section")

// renderConfig returns the config.json to write to the app dir: the seed
// unchanged when o is nil, else the seed with a logs_store block added
// as its last top-level key. The seed's own bytes are kept as they are
// (only the closing brace moves), so `diff` against the seed shows just
// the added block.
func renderConfig(seed []byte, o *logsOffload, appDir string) ([]byte, error) {
	if o == nil {
		return seed, nil
	}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(seed, &top); err != nil {
		return nil, fmt.Errorf("parse seed config.json: %w", err)
	}
	if top == nil {
		return nil, errors.New("parse seed config.json: not a JSON object")
	}
	if _, ok := top[logsStoreKey]; ok {
		return nil, errSeedHasLogsStore
	}

	block, err := json.MarshalIndent(logsStoreBlock(o, appDir), "  ", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal logs_store: %w", err)
	}

	// json.Unmarshal into a map succeeded, so the trimmed seed is one
	// object: it starts with '{' and ends with '}'.
	body := bytes.TrimRight(seed, " \t\r\n")
	body = bytes.TrimRight(body[:len(body)-1], " \t\r\n")

	var out bytes.Buffer
	out.Grow(len(seed) + len(block) + 32)
	out.Write(body)
	if len(top) > 0 {
		out.WriteByte(',')
	}
	out.WriteString("\n  \"" + logsStoreKey + "\": ")
	out.Write(block)
	out.WriteString("\n}\n")
	return out.Bytes(), nil
}
