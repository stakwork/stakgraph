package main

import (
	"context"
	"errors"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// dest is the offload destination, as the wrapper passes it.
type dest struct {
	Bucket, Prefix, Region, Endpoint string
	AccessKeyID, SecretAccessKey     string
}

// s3Client is the subset of *s3.Client this command uses, so the tests
// can run preflight and upload against an in-memory fake.
type s3Client interface {
	HeadBucket(context.Context, *s3.HeadBucketInput, ...func(*s3.Options)) (*s3.HeadBucketOutput, error)
	PutObject(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error)
	GetObjectTagging(context.Context, *s3.GetObjectTaggingInput, ...func(*s3.Options)) (*s3.GetObjectTaggingOutput, error)
	DeleteObject(context.Context, *s3.DeleteObjectInput, ...func(*s3.Options)) (*s3.DeleteObjectOutput, error)
	HeadObject(context.Context, *s3.HeadObjectInput, ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	CreateMultipartUpload(context.Context, *s3.CreateMultipartUploadInput, ...func(*s3.Options)) (*s3.CreateMultipartUploadOutput, error)
	UploadPart(context.Context, *s3.UploadPartInput, ...func(*s3.Options)) (*s3.UploadPartOutput, error)
	CompleteMultipartUpload(context.Context, *s3.CompleteMultipartUploadInput, ...func(*s3.Options)) (*s3.CompleteMultipartUploadOutput, error)
	AbortMultipartUpload(context.Context, *s3.AbortMultipartUploadInput, ...func(*s3.Options)) (*s3.AbortMultipartUploadOutput, error)
	ListMultipartUploads(context.Context, *s3.ListMultipartUploadsInput, ...func(*s3.Options)) (*s3.ListMultipartUploadsOutput, error)
}

// newS3Client builds the client the way Bifrost's object store does
// (framework/objectstore/s3.go NewS3ObjectStore at transports/v2.5.0):
// region when given, static credentials when the key pair is given, else
// the default chain, and BaseEndpoint + path-style when an endpoint is
// set (the wrapper sets force_path_style exactly then).
//
// forUpload turns the SDK's default request checksums off (only what an
// operation requires is sent) and the archive upload sets Content-MD5 on
// every part instead: the long-standing integrity check every
// S3-compatible store honours. The preflight client keeps the SDK
// defaults, so it sends exactly what Bifrost's own Put does.
func newS3Client(ctx context.Context, d dest, forUpload bool) (s3Client, error) {
	if d.Bucket == "" {
		return nil, errors.New("bucket is required")
	}
	if (d.AccessKeyID == "") != (d.SecretAccessKey == "") {
		return nil, errors.New("access key id and secret access key must be set together")
	}
	var opts []func(*awsconfig.LoadOptions) error
	if d.Region != "" {
		opts = append(opts, awsconfig.WithRegion(d.Region))
	}
	if d.AccessKeyID != "" {
		opts = append(opts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(d.AccessKeyID, d.SecretAccessKey, "")))
	}
	cfg, err := awsconfig.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return s3.NewFromConfig(cfg, func(o *s3.Options) {
		if d.Endpoint != "" {
			o.BaseEndpoint = aws.String(d.Endpoint)
			o.UsePathStyle = true
		}
		if forUpload {
			o.RequestChecksumCalculation = aws.RequestChecksumCalculationWhenRequired
		}
	}), nil
}
