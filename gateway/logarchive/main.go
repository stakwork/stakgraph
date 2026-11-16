// Command logs-archive is the gateway image's S3 helper for log payload
// offload (BIFROST_LOGS_S3_*). The wrapper (gateway/wrapper) runs it; it
// is not an entrypoint of its own.
//
//	logs-archive preflight -bucket B -prefix P [-region R] [-endpoint E]
//	logs-archive upload    -bucket B -prefix P [-region R] [-endpoint E] -dir D
//
// preflight proves the bucket works for Bifrost's hybrid log store before
// the wrapper turns offload on (and before it moves an old logs.db
// aside): HeadBucket (Bifrost's own boot check), a tagged PutObject (what
// every offloaded log does, so s3:PutObjectTagging is covered), and a
// GetObject of the same object (what every detail read does).
//
// upload archives one legacy directory (an old logs.db, with its -wal and
// -shm when present) to <prefix>/legacy/logs-<name>.tar.gz as a streamed,
// gzipped, multipart upload, verifies it, and only then deletes the local
// files. Any failure leaves the files where they are.
//
// The access key pair comes from BIFROST_LOGS_S3_ACCESS_KEY_ID /
// BIFROST_LOGS_S3_SECRET_ACCESS_KEY in the environment (never argv);
// without them the AWS default credential chain applies, as in Bifrost.
// Exit status 0 on success, 1 on failure (the reason is logged).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	logger := log.New(os.Stderr, "[logs-archive] ", log.LstdFlags|log.LUTC)
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: logs-archive preflight|upload -bucket B -prefix P [-region R] [-endpoint E] [-dir D]")
		os.Exit(2)
	}
	cmd := os.Args[1]

	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	var d dest
	fs.StringVar(&d.Bucket, "bucket", "", "bucket")
	fs.StringVar(&d.Prefix, "prefix", "bifrost", "key prefix")
	fs.StringVar(&d.Region, "region", "", "region (optional)")
	fs.StringVar(&d.Endpoint, "endpoint", "", "S3-compatible endpoint (optional; turns on path-style)")
	timeout := fs.Duration("timeout", 20*time.Second, "preflight: overall deadline")
	dir := fs.String("dir", "", "upload: the legacy directory to archive")
	_ = fs.Parse(os.Args[2:])
	d.AccessKeyID = os.Getenv("BIFROST_LOGS_S3_ACCESS_KEY_ID")
	d.SecretAccessKey = os.Getenv("BIFROST_LOGS_S3_SECRET_ACCESS_KEY")

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var err error
	switch cmd {
	case "preflight":
		ctx, cancel := context.WithTimeout(ctx, *timeout)
		defer cancel()
		var c s3Client
		if c, err = newS3Client(ctx, d, false); err == nil {
			err = preflight(ctx, c, d, logger)
		}
	case "upload":
		if *dir == "" {
			err = fmt.Errorf("-dir is required")
			break
		}
		var c s3Client
		if c, err = newS3Client(ctx, d, true); err == nil {
			err = archiveDir(ctx, c, d, *dir, logger)
		}
	default:
		err = fmt.Errorf("unknown command %q", cmd)
	}
	if err != nil {
		logger.Printf("%s failed: %v", cmd, err)
		os.Exit(1)
	}
}
