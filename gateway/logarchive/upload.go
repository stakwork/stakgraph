package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/md5"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

const (
	// minPartSize is S3's floor for every part but the last (5 MiB),
	// rounded up to a size that keeps the part count low.
	minPartSize = 16 << 20
	// maxPartSize is S3's ceiling for one part.
	maxPartSize = 5 << 30
	// maxParts is S3's ceiling for one upload.
	maxParts = 10000
	// partBudget is the part count the size is chosen for, leaving
	// headroom under maxParts for gzip growth on incompressible data.
	partBudget = 9000
	// progressEvery is how often a running upload logs its progress.
	progressEvery = 30 * time.Second
)

// partSizeFor picks the part size for an archive of `total` uncompressed
// bytes: never more than partBudget parts even if gzip saves nothing (a
// 133 GB logs.db gets 16 MiB parts, 500 GB 54 MiB, 1 TB 108 MiB).
// One part is held in memory at a time.
func partSizeFor(total int64) int64 {
	// tar headers and gzip framing add a little; 1% + 1 MiB covers it.
	need := (total + total/100 + 1<<20 + partBudget - 1) / partBudget
	size := int64(minPartSize)
	if need > size {
		size = (need + 1<<20 - 1) &^ (1<<20 - 1) // round up to MiB
	}
	if size > maxPartSize {
		size = maxPartSize
	}
	return size
}

// archiveKey is where a legacy directory's archive goes.
func archiveKey(prefix, dir string) string {
	return fmt.Sprintf("%s/legacy/logs-%s.tar.gz", prefix, filepath.Base(dir))
}

// fileSum describes one archived file, for the manifest.
type fileSum struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// manifest is written next to the archive (<key>.manifest.json) once the
// archive is verified: what is in it, so a restore can be checked.
type manifest struct {
	Archive    string    `json:"archive"`
	Files      []fileSum `json:"files"`
	GzipBytes  int64     `json:"gzip_bytes"`
	Parts      int       `json:"parts"`
	PartSize   int64     `json:"part_size"`
	UploadedAt string    `json:"uploaded_at"`
}

// legacyFiles lists the regular files in dir, the SQLite set first
// (logs.db, -wal, -shm), then anything else by name.
func legacyFiles(dir string) ([]string, int64, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, 0, err
	}
	rank := map[string]int{"logs.db": 0, "logs.db-wal": 1, "logs.db-shm": 2}
	var names []string
	var total int64
	for _, e := range ents {
		if !e.Type().IsRegular() {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			return nil, 0, err
		}
		names = append(names, e.Name())
		total += fi.Size()
	}
	sort.Slice(names, func(i, j int) bool {
		ri, iok := rank[names[i]]
		rj, jok := rank[names[j]]
		switch {
		case iok && jok:
			return ri < rj
		case iok != jok:
			return iok
		default:
			return names[i] < names[j]
		}
	})
	return names, total, nil
}

// archiveDir uploads dir as <prefix>/legacy/logs-<base>.tar.gz and, once
// the upload is verified, deletes the local files and the directory.
//
// The stream is tar (entries under logs-<base>/, so extracting never
// lands on a live logs.db) inside gzip (BestSpeed), cut into parts as it
// is produced: nothing is staged on disk and one part is in memory.
// Every part carries Content-MD5, so the store rejects a corrupted part.
// "Verified" means: every part accepted, CompleteMultipartUpload
// succeeded, and HeadObject reports exactly the bytes sent. Only then are
// local files removed. On any failure the upload is aborted and nothing
// local is touched, so the next run starts over.
func archiveDir(ctx context.Context, c s3Client, d dest, dir string, logger *log.Logger) error {
	names, total, err := legacyFiles(dir)
	if err != nil {
		return err
	}
	if len(names) == 0 {
		logger.Printf("%s is empty; removing it", dir)
		return os.Remove(dir)
	}
	key := archiveKey(d.Prefix, dir)
	partSize := partSizeFor(total)
	abortStale(ctx, c, d.Bucket, key, logger)

	create, err := c.CreateMultipartUpload(ctx, &s3.CreateMultipartUploadInput{
		Bucket:      aws.String(d.Bucket),
		Key:         aws.String(key),
		ContentType: aws.String("application/gzip"),
		Metadata: map[string]string{
			"source":             "stakgraph-gateway",
			"uncompressed-bytes": strconv.FormatInt(total, 10),
		},
	})
	if err != nil {
		return fmt.Errorf("create multipart upload s3://%s/%s: %w", d.Bucket, key, err)
	}
	uploadID := aws.ToString(create.UploadId)
	logger.Printf("archiving %s (%d files, %d bytes) to s3://%s/%s in %d MiB parts",
		dir, len(names), total, d.Bucket, key, partSize>>20)

	abort := func(cause error) error {
		// A fresh context: ctx may be the reason we are aborting.
		actx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if _, err := c.AbortMultipartUpload(actx, &s3.AbortMultipartUploadInput{
			Bucket: aws.String(d.Bucket), Key: aws.String(key), UploadId: aws.String(uploadID),
		}); err != nil {
			logger.Printf("warning: abort of upload %s failed (s3:AbortMultipartUpload; a lifecycle rule "+
				"for incomplete multipart uploads cleans it up): %v", uploadID, err)
		}
		return cause
	}

	pr, pw := io.Pipe()
	sums := make(chan []fileSum, 1)
	go func() {
		s, err := writeTarGz(pw, dir, names)
		sums <- s
		_ = pw.CloseWithError(err)
	}()

	parts, sent, err := uploadParts(ctx, c, d.Bucket, key, uploadID, pr, partSize, total, logger)
	if err != nil {
		_ = pr.CloseWithError(err) // unblock the writer
		<-sums
		return abort(err)
	}
	files := <-sums

	if _, err := c.CompleteMultipartUpload(ctx, &s3.CompleteMultipartUploadInput{
		Bucket:          aws.String(d.Bucket),
		Key:             aws.String(key),
		UploadId:        aws.String(uploadID),
		MultipartUpload: &types.CompletedMultipartUpload{Parts: parts},
	}); err != nil {
		return abort(fmt.Errorf("complete multipart upload: %w", err))
	}
	head, err := c.HeadObject(ctx, &s3.HeadObjectInput{Bucket: aws.String(d.Bucket), Key: aws.String(key)})
	if err != nil {
		return fmt.Errorf("verify s3://%s/%s: head object: %w (local files kept)", d.Bucket, key, err)
	}
	if got := aws.ToInt64(head.ContentLength); got != sent {
		return fmt.Errorf("verify s3://%s/%s: stored %d bytes, sent %d (local files kept)", d.Bucket, key, got, sent)
	}

	m := manifest{
		Archive: key, Files: files, GzipBytes: sent, Parts: len(parts), PartSize: partSize,
		UploadedAt: time.Now().UTC().Format(time.RFC3339),
	}
	mb, _ := json.MarshalIndent(m, "", "  ")
	if _, err := c.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(d.Bucket), Key: aws.String(key + ".manifest.json"),
		Body: bytes.NewReader(mb), ContentType: aws.String("application/json"),
	}); err != nil {
		// The archive itself is verified; the manifest is a convenience.
		logger.Printf("warning: manifest not written (the archive is complete and verified): %v", err)
	}

	for _, n := range names {
		if err := os.Remove(filepath.Join(dir, n)); err != nil {
			return fmt.Errorf("archive verified, but removing %s failed: %w", n, err)
		}
	}
	if err := os.Remove(dir); err != nil {
		return fmt.Errorf("archive verified, but removing %s failed: %w", dir, err)
	}
	logger.Printf("archived %s to s3://%s/%s (%d -> %d bytes, %d parts, verified); deleted the local copy",
		dir, d.Bucket, key, total, sent, len(parts))
	return nil
}

// abortStale aborts earlier, unfinished uploads to key: a container that
// was killed mid-upload leaves one behind, and parts of an incomplete
// upload are billed until aborted. Failures are logged only (needs
// s3:ListBucketMultipartUploads and s3:AbortMultipartUpload; a bucket
// lifecycle rule is the backstop).
func abortStale(ctx context.Context, c s3Client, bucket, key string, logger *log.Logger) {
	out, err := c.ListMultipartUploads(ctx, &s3.ListMultipartUploadsInput{
		Bucket: aws.String(bucket), Prefix: aws.String(key),
	})
	if err != nil {
		logger.Printf("warning: could not list earlier uploads to %s (s3:ListBucketMultipartUploads): %v", key, err)
		return
	}
	for _, u := range out.Uploads {
		if aws.ToString(u.Key) != key {
			continue
		}
		if _, err := c.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
			Bucket: aws.String(bucket), Key: u.Key, UploadId: u.UploadId,
		}); err != nil {
			logger.Printf("warning: could not abort earlier upload %s: %v", aws.ToString(u.UploadId), err)
			continue
		}
		logger.Printf("aborted an earlier, unfinished upload %s of %s", aws.ToString(u.UploadId), key)
	}
}

// uploadParts reads r in partSize chunks and uploads each with
// Content-MD5. It returns the completed parts and the bytes sent. The SDK
// retries transient failures of each call itself.
func uploadParts(ctx context.Context, c s3Client, bucket, key, uploadID string, r io.Reader,
	partSize, total int64, logger *log.Logger) ([]types.CompletedPart, int64, error) {
	buf := make([]byte, partSize)
	var parts []types.CompletedPart
	var sent int64
	start, lastLog := time.Now(), time.Now()
	for n := int32(1); ; n++ {
		read, rerr := io.ReadFull(r, buf)
		if read > 0 {
			if n > maxParts {
				return nil, sent, fmt.Errorf("archive needs more than %d parts of %d bytes", maxParts, partSize)
			}
			sum := md5.Sum(buf[:read])
			out, err := c.UploadPart(ctx, &s3.UploadPartInput{
				Bucket:        aws.String(bucket),
				Key:           aws.String(key),
				UploadId:      aws.String(uploadID),
				PartNumber:    aws.Int32(n),
				Body:          bytes.NewReader(buf[:read]),
				ContentLength: aws.Int64(int64(read)),
				ContentMD5:    aws.String(base64.StdEncoding.EncodeToString(sum[:])),
			})
			if err != nil {
				return nil, sent, fmt.Errorf("upload part %d: %w", n, err)
			}
			parts = append(parts, types.CompletedPart{ETag: out.ETag, PartNumber: aws.Int32(n)})
			sent += int64(read)
			if time.Since(lastLog) >= progressEvery {
				lastLog = time.Now()
				logger.Printf("archiving %s: %d parts, %d MiB sent, %s elapsed (source %d MiB)",
					key, n, sent>>20, time.Since(start).Round(time.Second), total>>20)
			}
		}
		if errors.Is(rerr, io.EOF) || errors.Is(rerr, io.ErrUnexpectedEOF) {
			return parts, sent, nil
		}
		if rerr != nil {
			return nil, sent, fmt.Errorf("read archive stream: %w", rerr)
		}
	}
}

// writeTarGz writes names (files in dir) to w as a gzipped tar with the
// entries under logs-<base(dir)>/, and returns each file's size and
// SHA-256 as read.
func writeTarGz(w io.Writer, dir string, names []string) ([]fileSum, error) {
	gz, _ := gzip.NewWriterLevel(w, gzip.BestSpeed)
	tw := tar.NewWriter(gz)
	top := "logs-" + filepath.Base(dir)
	var sums []fileSum
	for _, name := range names {
		s, err := addFile(tw, filepath.Join(dir, name), top+"/"+name)
		if err != nil {
			return sums, err
		}
		s.Name = name
		sums = append(sums, s)
	}
	if err := tw.Close(); err != nil {
		return sums, err
	}
	return sums, gz.Close()
}

func addFile(tw *tar.Writer, path, name string) (fileSum, error) {
	f, err := os.Open(path)
	if err != nil {
		return fileSum{}, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return fileSum{}, err
	}
	if err := tw.WriteHeader(&tar.Header{
		Name: name, Mode: 0o644, Size: fi.Size(), ModTime: fi.ModTime(), Typeflag: tar.TypeReg,
	}); err != nil {
		return fileSum{}, err
	}
	h := sha256.New()
	if _, err := io.CopyN(io.MultiWriter(tw, h), f, fi.Size()); err != nil {
		return fileSum{}, fmt.Errorf("read %s: %w", path, err)
	}
	return fileSum{Bytes: fi.Size(), SHA256: hex.EncodeToString(h.Sum(nil))}, nil
}
