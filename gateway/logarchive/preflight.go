package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// preflightTag marks the probe object; it is also what the tag read-back
// looks for.
const preflightTagKey, preflightTagValue = "source", "stakgraph-gateway-preflight"

// preflight proves the bucket takes what Bifrost's hybrid log store will
// send it. Hard requirements (an error turns offload off for this boot):
//
//   - HeadBucket: Bifrost's own Ping at boot (s3:ListBucket).
//   - PutObject with tags: every offloaded log is a tagged Put
//     (s3:PutObject + s3:PutObjectTagging; S3 refuses a tagged Put
//     without the latter, and without this check that refusal would only
//     show up as a dropped upload per log, after the row was stripped).
//   - GetObject of the probe, byte-equal: every detail read (s3:GetObject).
//
// Reading the tags back (s3:GetObjectTagging) and deleting the probe
// (s3:DeleteObject) are logged when they fail but do not fail the check:
// Bifrost needs neither to offload and read logs.
func preflight(ctx context.Context, c s3Client, d dest, logger *log.Logger) error {
	if _, err := c.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(d.Bucket)}); err != nil {
		return fmt.Errorf("head bucket %s (needs s3:ListBucket): %w", d.Bucket, err)
	}

	host, _ := os.Hostname()
	key := fmt.Sprintf("%s/preflight/probe-%s-%d.json", d.Prefix, host, time.Now().UnixNano())
	body := []byte(fmt.Sprintf(`{"probe":"stakgraph-gateway logs offload preflight","at":%q}`,
		time.Now().UTC().Format(time.RFC3339Nano)))
	if _, err := c.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(d.Bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(body),
		ContentType: aws.String("application/json"),
		Tagging:     aws.String(preflightTagKey + "=" + preflightTagValue),
	}); err != nil {
		return fmt.Errorf("tagged put s3://%s/%s (needs s3:PutObject and s3:PutObjectTagging): %w", d.Bucket, key, err)
	}

	got, err := c.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(d.Bucket), Key: aws.String(key)})
	if err != nil {
		return fmt.Errorf("get s3://%s/%s (needs s3:GetObject): %w", d.Bucket, key, err)
	}
	read, err := io.ReadAll(got.Body)
	_ = got.Body.Close()
	if err != nil {
		return fmt.Errorf("read s3://%s/%s: %w", d.Bucket, key, err)
	}
	if !bytes.Equal(read, body) {
		return fmt.Errorf("s3://%s/%s read back %d bytes that differ from the %d written", d.Bucket, key, len(read), len(body))
	}

	extra := ""
	if tags, err := c.GetObjectTagging(ctx, &s3.GetObjectTaggingInput{Bucket: aws.String(d.Bucket), Key: aws.String(key)}); err != nil {
		logger.Printf("preflight warning: could not read the probe's tags back (s3:GetObjectTagging): %v", err)
	} else if !hasTag(tags, preflightTagKey, preflightTagValue) {
		logger.Printf("preflight warning: the store accepted the tagged put but kept no tags")
	} else {
		extra = ", tags read back"
	}
	if _, err := c.DeleteObject(ctx, &s3.DeleteObjectInput{Bucket: aws.String(d.Bucket), Key: aws.String(key)}); err != nil {
		logger.Printf("preflight warning: could not delete the probe s3://%s/%s (s3:DeleteObject): %v", d.Bucket, key, err)
	}
	logger.Printf("preflight ok: s3://%s/%s/ (head bucket, tagged put, get%s)", d.Bucket, d.Prefix, extra)
	return nil
}

func hasTag(out *s3.GetObjectTaggingOutput, k, v string) bool {
	for _, t := range out.TagSet {
		if aws.ToString(t.Key) == k && aws.ToString(t.Value) == v {
			return true
		}
	}
	return false
}
