package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/md5"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// fakeS3 is an in-memory S3 with just enough semantics: objects with
// tags, multipart uploads that check Content-MD5 per part, and switches
// to deny an operation the way a missing IAM permission does.
type fakeS3 struct {
	mu       sync.Mutex
	objects  map[string][]byte
	tags     map[string]string // key -> raw Tagging
	uploads  map[string]*fakeUpload
	nextID   int
	deny     map[string]bool // operation name -> AccessDenied
	failPart int32           // UploadPart fails for this part number
	shortLen bool            // HeadObject under-reports the size
	calls    []string
}

type fakeUpload struct {
	key   string
	parts map[int32][]byte
}

var errDenied = errors.New("AccessDenied")

func newFake() *fakeS3 {
	return &fakeS3{objects: map[string][]byte{}, tags: map[string]string{}, uploads: map[string]*fakeUpload{}, deny: map[string]bool{}}
}

func (f *fakeS3) op(name string) error {
	f.calls = append(f.calls, name)
	if f.deny[name] {
		return fmt.Errorf("%s: %w", name, errDenied)
	}
	return nil
}

func (f *fakeS3) HeadBucket(_ context.Context, _ *s3.HeadBucketInput, _ ...func(*s3.Options)) (*s3.HeadBucketOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return &s3.HeadBucketOutput{}, f.op("HeadBucket")
}

func (f *fakeS3) PutObject(_ context.Context, in *s3.PutObjectInput, _ ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("PutObject"); err != nil {
		return nil, err
	}
	if in.Tagging != nil {
		if err := f.op("PutObjectTagging"); err != nil {
			return nil, err
		}
	}
	b, _ := io.ReadAll(in.Body)
	f.objects[*in.Key] = b
	if in.Tagging != nil {
		f.tags[*in.Key] = *in.Tagging
	}
	return &s3.PutObjectOutput{}, nil
}

func (f *fakeS3) GetObject(_ context.Context, in *s3.GetObjectInput, _ ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("GetObject"); err != nil {
		return nil, err
	}
	b, ok := f.objects[*in.Key]
	if !ok {
		return nil, errors.New("NoSuchKey")
	}
	return &s3.GetObjectOutput{Body: io.NopCloser(bytes.NewReader(b))}, nil
}

func (f *fakeS3) GetObjectTagging(_ context.Context, in *s3.GetObjectTaggingInput, _ ...func(*s3.Options)) (*s3.GetObjectTaggingOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("GetObjectTagging"); err != nil {
		return nil, err
	}
	out := &s3.GetObjectTaggingOutput{}
	q, _ := url.ParseQuery(f.tags[*in.Key])
	for k, vs := range q {
		out.TagSet = append(out.TagSet, types.Tag{Key: aws.String(k), Value: aws.String(vs[0])})
	}
	return out, nil
}

func (f *fakeS3) DeleteObject(_ context.Context, in *s3.DeleteObjectInput, _ ...func(*s3.Options)) (*s3.DeleteObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("DeleteObject"); err != nil {
		return nil, err
	}
	delete(f.objects, *in.Key)
	return &s3.DeleteObjectOutput{}, nil
}

func (f *fakeS3) HeadObject(_ context.Context, in *s3.HeadObjectInput, _ ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("HeadObject"); err != nil {
		return nil, err
	}
	b, ok := f.objects[*in.Key]
	if !ok {
		return nil, errors.New("NotFound")
	}
	n := int64(len(b))
	if f.shortLen {
		n--
	}
	return &s3.HeadObjectOutput{ContentLength: aws.Int64(n)}, nil
}

func (f *fakeS3) CreateMultipartUpload(_ context.Context, in *s3.CreateMultipartUploadInput, _ ...func(*s3.Options)) (*s3.CreateMultipartUploadOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("CreateMultipartUpload"); err != nil {
		return nil, err
	}
	f.nextID++
	id := fmt.Sprintf("upload-%d", f.nextID)
	f.uploads[id] = &fakeUpload{key: *in.Key, parts: map[int32][]byte{}}
	return &s3.CreateMultipartUploadOutput{UploadId: aws.String(id)}, nil
}

func (f *fakeS3) UploadPart(_ context.Context, in *s3.UploadPartInput, _ ...func(*s3.Options)) (*s3.UploadPartOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("UploadPart"); err != nil {
		return nil, err
	}
	if f.failPart != 0 && *in.PartNumber == f.failPart {
		return nil, errors.New("InternalError")
	}
	u, ok := f.uploads[*in.UploadId]
	if !ok {
		return nil, errors.New("NoSuchUpload")
	}
	b, _ := io.ReadAll(in.Body)
	sum := md5.Sum(b)
	if in.ContentMD5 == nil || *in.ContentMD5 != base64.StdEncoding.EncodeToString(sum[:]) {
		return nil, errors.New("BadDigest")
	}
	u.parts[*in.PartNumber] = b
	return &s3.UploadPartOutput{ETag: aws.String(hex.EncodeToString(sum[:]))}, nil
}

func (f *fakeS3) CompleteMultipartUpload(_ context.Context, in *s3.CompleteMultipartUploadInput, _ ...func(*s3.Options)) (*s3.CompleteMultipartUploadOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("CompleteMultipartUpload"); err != nil {
		return nil, err
	}
	u, ok := f.uploads[*in.UploadId]
	if !ok {
		return nil, errors.New("NoSuchUpload")
	}
	var buf bytes.Buffer
	for i, p := range in.MultipartUpload.Parts {
		if *p.PartNumber != int32(i+1) {
			return nil, errors.New("InvalidPartOrder")
		}
		b := u.parts[*p.PartNumber]
		if i < len(in.MultipartUpload.Parts)-1 && len(b) < 5<<20 {
			return nil, errors.New("EntityTooSmall")
		}
		buf.Write(b)
	}
	f.objects[u.key] = buf.Bytes()
	delete(f.uploads, *in.UploadId)
	return &s3.CompleteMultipartUploadOutput{}, nil
}

func (f *fakeS3) AbortMultipartUpload(_ context.Context, in *s3.AbortMultipartUploadInput, _ ...func(*s3.Options)) (*s3.AbortMultipartUploadOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("AbortMultipartUpload"); err != nil {
		return nil, err
	}
	delete(f.uploads, *in.UploadId)
	return &s3.AbortMultipartUploadOutput{}, nil
}

func (f *fakeS3) ListMultipartUploads(_ context.Context, in *s3.ListMultipartUploadsInput, _ ...func(*s3.Options)) (*s3.ListMultipartUploadsOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.op("ListMultipartUploads"); err != nil {
		return nil, err
	}
	out := &s3.ListMultipartUploadsOutput{}
	for id, u := range f.uploads {
		if strings.HasPrefix(u.key, aws.ToString(in.Prefix)) {
			out.Uploads = append(out.Uploads, types.MultipartUpload{Key: aws.String(u.key), UploadId: aws.String(id)})
		}
	}
	return out, nil
}

// ─── helpers ─────────────────────────────────────────────────────────

func quiet() *log.Logger { return log.New(io.Discard, "", 0) }

var testDest = dest{Bucket: "fleet", Prefix: "swarm1"}

// legacyDir makes <tmp>/logs-legacy/<name> holding files of the given
// sizes (random bytes, so gzip cannot shrink them).
func legacyDir(t *testing.T, name string, sizes map[string]int) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "logs-legacy", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for n, size := range sizes {
		b := make([]byte, size)
		_, _ = rand.Read(b)
		if err := os.WriteFile(filepath.Join(dir, n), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func fileSHA(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// untar returns name -> sha256 for every entry of a .tar.gz.
func untar(t *testing.T, b []byte) map[string]string {
	t.Helper()
	gz, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("not gzip: %v", err)
	}
	tr := tar.NewReader(gz)
	out := map[string]string{}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatalf("tar: %v", err)
		}
		d := sha256.New()
		if _, err := io.Copy(d, tr); err != nil {
			t.Fatal(err)
		}
		out[h.Name] = hex.EncodeToString(d.Sum(nil))
	}
}

// ─── upload ──────────────────────────────────────────────────────────

func TestPartSizeFor(t *testing.T) {
	for _, tc := range []struct {
		total int64
		want  int64
	}{
		{0, 16 << 20},
		{1 << 30, 16 << 20},
		{133e9, 16 << 20},
		{500e9, 54 << 20},
		{1e12, 108 << 20},
		{5e12, 536 << 20},
		{100e12, maxPartSize},
	} {
		got := partSizeFor(tc.total)
		if got != tc.want {
			t.Errorf("partSizeFor(%d) = %d MiB, want %d MiB", tc.total, got>>20, tc.want>>20)
		}
		if got < maxPartSize && (tc.total+tc.total/100)/got >= partBudget {
			t.Errorf("partSizeFor(%d) = %d leaves no headroom", tc.total, got)
		}
	}
}

func TestArchiveDirRoundTrip(t *testing.T) {
	// 40 MiB of incompressible data: three parts of 16 MiB.
	dir := legacyDir(t, "20261116T041000Z", map[string]int{
		"logs.db": 36 << 20, "logs.db-wal": 4 << 20, "logs.db-shm": 32 << 10,
	})
	want := map[string]string{}
	for _, n := range []string{"logs.db", "logs.db-wal", "logs.db-shm"} {
		want["logs-20261116T041000Z/"+n] = fileSHA(t, filepath.Join(dir, n))
	}

	f := newFake()
	if err := archiveDir(context.Background(), f, testDest, dir, quiet()); err != nil {
		t.Fatalf("archiveDir: %v", err)
	}
	key := "swarm1/legacy/logs-20261116T041000Z.tar.gz"
	obj, ok := f.objects[key]
	if !ok {
		t.Fatalf("no object at %s; have %v", key, keys(f.objects))
	}
	if got := untar(t, obj); !mapsEqual(got, want) {
		t.Errorf("archive contents = %v, want %v", got, want)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("local dir still there after a verified upload: %v", err)
	}
	var m manifest
	if err := json.Unmarshal(f.objects[key+".manifest.json"], &m); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if m.GzipBytes != int64(len(obj)) || m.Parts != 3 || len(m.Files) != 3 || m.Files[0].Name != "logs.db" {
		t.Errorf("manifest = %+v", m)
	}
	if m.Files[0].SHA256 != want["logs-20261116T041000Z/logs.db"] {
		t.Errorf("manifest sha256 of logs.db = %s, want %s", m.Files[0].SHA256, want["logs-20261116T041000Z/logs.db"])
	}
	if len(f.uploads) != 0 {
		t.Errorf("uploads left open: %d", len(f.uploads))
	}
}

func TestArchiveDirAbortsStaleUploads(t *testing.T) {
	dir := legacyDir(t, "T1", map[string]int{"logs.db": 1 << 20})
	f := newFake()
	key := archiveKey(testDest.Prefix, dir)
	// A killed earlier run left an upload behind, plus one for another key.
	f.uploads["stale"] = &fakeUpload{key: key, parts: map[int32][]byte{1: []byte("x")}}
	f.uploads["other"] = &fakeUpload{key: key + "-other", parts: map[int32][]byte{}}
	if err := archiveDir(context.Background(), f, testDest, dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.uploads["stale"]; ok {
		t.Error("stale upload for the same key not aborted")
	}
	if _, ok := f.uploads["other"]; !ok {
		t.Error("upload for another key was aborted")
	}
}

func TestArchiveDirFailuresKeepLocalFiles(t *testing.T) {
	for _, tc := range []struct {
		name  string
		setup func(*fakeS3)
	}{
		{"create denied", func(f *fakeS3) { f.deny["CreateMultipartUpload"] = true }},
		{"part fails", func(f *fakeS3) { f.failPart = 2 }},
		{"complete fails", func(f *fakeS3) { f.deny["CompleteMultipartUpload"] = true }},
		{"head fails", func(f *fakeS3) { f.deny["HeadObject"] = true }},
		{"size mismatch", func(f *fakeS3) { f.shortLen = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := legacyDir(t, "T2", map[string]int{"logs.db": 20 << 20})
			sha := fileSHA(t, filepath.Join(dir, "logs.db"))
			f := newFake()
			tc.setup(f)
			if err := archiveDir(context.Background(), f, testDest, dir, quiet()); err == nil {
				t.Fatal("want an error")
			}
			if got := fileSHA(t, filepath.Join(dir, "logs.db")); got != sha {
				t.Error("local logs.db changed")
			}
			if tc.name == "part fails" && len(f.uploads) != 0 {
				t.Error("failed upload not aborted")
			}
		})
	}
}

func TestArchiveDirManifestFailureStillDeletes(t *testing.T) {
	// The archive is verified before the manifest; a refused manifest
	// put is a warning, and the verified archive's local copy goes.
	dir := legacyDir(t, "T3", map[string]int{"logs.db": 1 << 20})
	f := newFake()
	f.deny["PutObject"] = true
	if err := archiveDir(context.Background(), f, testDest, dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Error("local copy kept")
	}
}

func TestArchiveDirCancelAborts(t *testing.T) {
	dir := legacyDir(t, "T4", map[string]int{"logs.db": 40 << 20})
	f := newFake()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := archiveDir(ctx, cancelingClient{f}, testDest, dir, quiet()); err == nil {
		t.Fatal("want an error")
	}
	if len(f.uploads) != 0 {
		t.Error("cancelled upload not aborted")
	}
	if _, err := os.Stat(filepath.Join(dir, "logs.db")); err != nil {
		t.Errorf("local file gone: %v", err)
	}
}

// cancelingClient fails UploadPart when ctx is done, as the SDK does.
type cancelingClient struct{ *fakeS3 }

func (c cancelingClient) UploadPart(ctx context.Context, in *s3.UploadPartInput, o ...func(*s3.Options)) (*s3.UploadPartOutput, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return c.fakeS3.UploadPart(ctx, in, o...)
}

func (c cancelingClient) ListMultipartUploads(ctx context.Context, in *s3.ListMultipartUploadsInput, o ...func(*s3.Options)) (*s3.ListMultipartUploadsOutput, error) {
	return c.fakeS3.ListMultipartUploads(ctx, in, o...)
}

func TestArchiveDirEmpty(t *testing.T) {
	dir := legacyDir(t, "T5", nil)
	if err := archiveDir(context.Background(), newFake(), testDest, dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Error("empty dir not removed")
	}
}

func TestLegacyFilesOrder(t *testing.T) {
	dir := legacyDir(t, "T6", map[string]int{"logs.db-shm": 1, "zzz": 1, "logs.db": 1, "aaa": 1, "logs.db-wal": 1})
	names, total, err := legacyFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"logs.db", "logs.db-wal", "logs.db-shm", "aaa", "zzz"}; strings.Join(names, ",") != strings.Join(want, ",") || total != 5 {
		t.Errorf("got %v (%d)", names, total)
	}
}

// ─── preflight ───────────────────────────────────────────────────────

func TestPreflight(t *testing.T) {
	for _, tc := range []struct {
		deny    string
		wantErr string // "" = passes
	}{
		{"", ""},
		{"HeadBucket", "s3:ListBucket"},
		{"PutObject", "s3:PutObject"},
		{"PutObjectTagging", "s3:PutObjectTagging"},
		{"GetObject", "s3:GetObject"},
		{"GetObjectTagging", ""}, // warning only
		{"DeleteObject", ""},     // warning only
	} {
		t.Run("deny "+tc.deny, func(t *testing.T) {
			f := newFake()
			if tc.deny != "" {
				f.deny[tc.deny] = true
			}
			err := preflight(context.Background(), f, testDest, quiet())
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("want ok, got %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want it to name %s", err, tc.wantErr)
			}
			// The probe lives under the prefix, and is gone when it can be.
			for k := range f.objects {
				if !strings.HasPrefix(k, "swarm1/preflight/") {
					t.Errorf("object outside the prefix: %s", k)
				}
				if tc.deny == "" {
					t.Errorf("probe %s not deleted", k)
				}
			}
		})
	}
}

func keys(m map[string][]byte) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
