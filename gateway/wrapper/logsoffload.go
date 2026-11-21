package main

// Log payload offload to S3-compatible object storage.
//
// Bifrost keeps every request's full payload (input history, output,
// params, tools, raw request/response, ...) in one SQLite file,
// <app-dir>/logs.db, and nothing ever shrinks it. Bifrost
// transports/v2.5.0 can instead run a "hybrid" log store: each log's
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
	"io"
	"path/filepath"
	"strconv"
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

// offloadRetentionDays is client.log_retention_days in the config
// written with offload on. The plugin dashboard's longest window is 30
// days, and with payloads in the bucket, logs.db only needs the index rows
// for that long (they still add up over a year on a busy swarm). With
// offload off the seed's value (365) stays: nothing is deleted that has
// not been archived. A seed already at or below 30 keeps its own value.
//
// Bifrost reconciles client config by hash (lib.loadClientConfig): a
// changed value in config.json wins over the one stored in config.db on
// the next boot, in both directions.
const offloadRetentionDays = 30

// renderConfig returns the config.json to write to the app dir: the seed
// unchanged when o is nil. With offload on: the seed with a logs_store
// block added as its last top-level key and client.log_retention_days
// lowered to offloadRetentionDays. Top-level keys keep their order and
// every untouched value keeps its bytes, so `diff` against the seed shows
// just those two changes.
func renderConfig(seed []byte, o *logsOffload, appDir string) ([]byte, error) {
	if o == nil {
		return seed, nil
	}

	members, err := parseObject(seed)
	if err != nil {
		return nil, fmt.Errorf("parse seed config.json: %w", err)
	}
	if indexOf(members, logsStoreKey) >= 0 {
		return nil, errSeedHasLogsStore
	}

	ci := indexOf(members, "client")
	var client json.RawMessage
	if ci >= 0 {
		client = members[ci].raw
	}
	client, err = withRetention(client, offloadRetentionDays)
	if err != nil {
		return nil, fmt.Errorf("seed config.json client: %w", err)
	}
	if ci >= 0 {
		members[ci].raw = client
	} else {
		members = append(members, member{key: "client", raw: client})
	}

	block, err := json.MarshalIndent(logsStoreBlock(o, appDir), "  ", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal logs_store: %w", err)
	}
	members = append(members, member{key: logsStoreKey, raw: block})
	return append(emitObject(members, ""), '\n'), nil
}

// withRetention returns the client object with log_retention_days set to
// days, unless it is already a positive value at or below days (then the
// object is returned untouched).
func withRetention(client json.RawMessage, days int) (json.RawMessage, error) {
	var members []member
	if client != nil {
		var err error
		if members, err = parseObject(client); err != nil {
			return nil, err
		}
	}
	want := json.RawMessage(strconv.Itoa(days))
	if i := indexOf(members, "log_retention_days"); i >= 0 {
		var cur int
		if json.Unmarshal(members[i].raw, &cur) == nil && cur > 0 && cur <= days {
			return client, nil
		}
		members[i].raw = want
	} else {
		members = append(members, member{key: "log_retention_days", raw: want})
	}
	return emitObject(members, "  "), nil
}

// member is one key of a JSON object, its value kept as the raw bytes it
// was written with.
type member struct {
	key string
	raw json.RawMessage
}

// parseObject splits a JSON object into its members, in order.
func parseObject(b []byte) ([]member, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return nil, errors.New("not a JSON object")
	}
	var out []member
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, err
		}
		k, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("unexpected token %v", kt)
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, err
		}
		out = append(out, member{key: k, raw: raw})
	}
	if _, err := dec.Token(); err != nil { // the closing brace
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("data after the JSON object")
	}
	return out, nil
}

// emitObject writes members as an object in the seed's layout: one key
// per line, two-space indent below indent, values as they were.
func emitObject(ms []member, indent string) []byte {
	if len(ms) == 0 {
		return []byte("{}")
	}
	var b bytes.Buffer
	b.WriteString("{\n")
	for i, m := range ms {
		k, _ := json.Marshal(m.key)
		b.WriteString(indent + "  ")
		b.Write(k)
		b.WriteString(": ")
		b.Write(m.raw)
		if i < len(ms)-1 {
			b.WriteByte(',')
		}
		b.WriteByte('\n')
	}
	b.WriteString(indent + "}")
	return b.Bytes()
}

func indexOf(ms []member, key string) int {
	for i, m := range ms {
		if m.key == key {
			return i
		}
	}
	return -1
}
