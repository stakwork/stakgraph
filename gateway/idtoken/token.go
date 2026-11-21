package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/sts"
)

const (
	defaultTokenFile = "/var/run/secrets/anthropic.com/token"
	defaultAudience  = "https://api.anthropic.com"
	defaultDuration  = 600 * time.Second
	defaultRefresh   = 240 * time.Second

	minDuration = 60 * time.Second
	maxDuration = 3600 * time.Second

	// mintTimeout bounds one STS call (credential resolution through
	// IMDS included).
	mintTimeout = 30 * time.Second

	backoffMin = 5 * time.Second
	backoffMax = 60 * time.Second
)

type config struct {
	Region    string
	File      string
	Audience  string
	Duration  time.Duration
	Refresh   time.Duration
	Algorithm string
}

// configFromEnv reads and validates the helper's settings.
func configFromEnv(getenv func(string) string) (config, error) {
	get := func(k, def string) string {
		if v := strings.TrimSpace(getenv(k)); v != "" {
			return v
		}
		return def
	}
	secs := func(k string, def time.Duration) (time.Duration, error) {
		v := strings.TrimSpace(getenv(k))
		if v == "" {
			return def, nil
		}
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("%s=%q: want a positive whole number of seconds", k, v)
		}
		return time.Duration(n) * time.Second, nil
	}

	c := config{
		Region:    get("AWS_REGION", get("AWS_DEFAULT_REGION", "")),
		File:      get("ANTHROPIC_IDENTITY_TOKEN_FILE", defaultTokenFile),
		Audience:  get("ANTHROPIC_IDENTITY_TOKEN_AUDIENCE", defaultAudience),
		Algorithm: strings.ToUpper(get("ANTHROPIC_IDENTITY_TOKEN_SIGNING_ALGORITHM", "RS256")),
	}
	var err error
	if c.Duration, err = secs("ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS", defaultDuration); err != nil {
		return c, err
	}
	if c.Refresh, err = secs("ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS", defaultRefresh); err != nil {
		return c, err
	}

	switch {
	case c.Region == "":
		return c, errors.New("AWS_REGION (or AWS_DEFAULT_REGION) is required: GetWebIdentityToken is served only by regional STS endpoints")
	case !filepath.IsAbs(c.File):
		return c, fmt.Errorf("ANTHROPIC_IDENTITY_TOKEN_FILE=%q must be an absolute path", c.File)
	case c.Algorithm != "RS256" && c.Algorithm != "ES384":
		return c, fmt.Errorf("ANTHROPIC_IDENTITY_TOKEN_SIGNING_ALGORITHM=%q: want RS256 or ES384", c.Algorithm)
	case c.Duration < minDuration || c.Duration > maxDuration:
		return c, fmt.Errorf("ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS=%d: want 60-3600", int(c.Duration/time.Second))
	case c.Refresh >= c.Duration:
		return c, fmt.Errorf("ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS=%d must be less than ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS=%d",
			int(c.Refresh/time.Second), int(c.Duration/time.Second))
	}
	return c, nil
}

// stsEndpoint is the regional STS endpoint for region.
func stsEndpoint(region string) string {
	if strings.HasPrefix(region, "cn-") {
		return "https://sts." + region + ".amazonaws.com.cn"
	}
	return "https://sts." + region + ".amazonaws.com"
}

// tokenSource mints one identity token.
type tokenSource interface {
	Token(ctx context.Context) (string, error)
}

type stsSource struct {
	client *sts.Client
	cfg    config
}

// newSTSSource builds the STS client: default credential chain (env,
// shared config, IMDS instance profile) and the explicit regional endpoint.
func newSTSSource(ctx context.Context, cfg config) (*stsSource, error) {
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(cfg.Region))
	if err != nil {
		return nil, fmt.Errorf("load AWS config: %w", err)
	}
	client := sts.NewFromConfig(awsCfg, func(o *sts.Options) {
		o.BaseEndpoint = aws.String(stsEndpoint(cfg.Region))
	})
	return &stsSource{client: client, cfg: cfg}, nil
}

func (s *stsSource) Token(ctx context.Context) (string, error) {
	out, err := s.client.GetWebIdentityToken(ctx, &sts.GetWebIdentityTokenInput{
		Audience:         []string{s.cfg.Audience},
		SigningAlgorithm: aws.String(s.cfg.Algorithm),
		DurationSeconds:  aws.Int32(int32(s.cfg.Duration / time.Second)),
	})
	if err != nil {
		return "", fmt.Errorf("sts:GetWebIdentityToken (%s): %w", stsEndpoint(s.cfg.Region), err)
	}
	tok := strings.TrimSpace(aws.ToString(out.WebIdentityToken))
	if tok == "" {
		return "", errors.New("sts:GetWebIdentityToken returned an empty token")
	}
	return tok, nil
}

// writeTokenFile replaces path with token atomically: temp file in the
// same directory (created if missing), fsync, rename. Mode 0600: bifrost
// runs as the same user as this helper. A reader sees the old token or
// the new one, never a torn file.
func writeTokenFile(path, token string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create %s: %w", dir, err)
	}
	tmp, err := os.CreateTemp(dir, ".token-*")
	if err != nil {
		return fmt.Errorf("create temp file in %s: %w", dir, err)
	}
	tmpPath := tmp.Name()
	fail := func(what string, err error) error {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("%s %s: %w", what, tmpPath, err)
	}
	if err := tmp.Chmod(0o600); err != nil {
		return fail("chmod", err)
	}
	if _, err := tmp.WriteString(token); err != nil {
		return fail("write", err)
	}
	if err := tmp.Sync(); err != nil {
		return fail("fsync", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close %s: %w", tmpPath, err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename %s -> %s: %w", tmpPath, path, err)
	}
	return nil
}

// mintOnce mints one token and writes it. The previous file is left
// alone on any failure.
func mintOnce(ctx context.Context, cfg config, src tokenSource, logger *log.Logger) error {
	mctx, cancel := context.WithTimeout(ctx, mintTimeout)
	defer cancel()
	tok, err := src.Token(mctx)
	if err != nil {
		return err
	}
	if err := writeTokenFile(cfg.File, tok); err != nil {
		return err
	}
	logger.Printf("wrote identity token to %s (audience %s, %s, lifetime %ds, next refresh in %ds)",
		cfg.File, cfg.Audience, cfg.Algorithm, int(cfg.Duration/time.Second), int(cfg.Refresh/time.Second))
	return nil
}

// backoff is the retry delay after the n-th (0-based) consecutive failure:
// doubling from backoffMin to backoffMax, and never longer than refresh.
func backoff(n int, refresh time.Duration) time.Duration {
	d := backoffMin
	for i := 0; i < n && d < backoffMax; i++ {
		d *= 2
	}
	if d > backoffMax {
		d = backoffMax
	}
	if d > refresh {
		d = refresh
	}
	return d
}

// sleepCtx waits d or until ctx ends; false means ctx ended.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// loop mints a token every cfg.Refresh until ctx ends. A failure is
// logged (the error never contains the token) and retried after a short
// backoff; the file keeps the previous token meanwhile. sleep is
// injectable for tests.
func loop(ctx context.Context, cfg config, src tokenSource, logger *log.Logger, sleep func(context.Context, time.Duration) bool) error {
	failures := 0
	for {
		wait := cfg.Refresh
		if err := mintOnce(ctx, cfg, src, logger); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			wait = backoff(failures, cfg.Refresh)
			failures++
			logger.Printf("ERROR: refresh failed (attempt %d), keeping the previous token file, retrying in %s: %v", failures, wait, err)
		} else {
			if failures > 0 {
				logger.Printf("refresh recovered after %d failure(s)", failures)
			}
			failures = 0
		}
		if !sleep(ctx, wait) {
			return ctx.Err()
		}
	}
}
