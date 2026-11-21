// Command id-token is the gateway image's helper for Anthropic workload
// identity federation from an EC2 instance profile. The wrapper
// (gateway/wrapper) runs it; it is not an entrypoint of its own.
//
//	id-token -once   mint one token, write it, exit (the wrapper runs this
//	                 at boot so a token exists before bifrost-http starts)
//	id-token         mint, write, and repeat every REFRESH seconds
//
// Bifrost does not call AWS. With anthropic_key_config.workload_identity
// and an identity_source of type "file" it re-reads the JWT from the file
// each time it needs a new Anthropic access token and POSTs it to
// https://api.anthropic.com/v1/oauth/token. This helper is what puts a
// JWT there: it calls sts:GetWebIdentityToken (AWS outbound identity
// federation) with the default AWS credential chain (the EC2 instance
// profile through IMDS) against the REGIONAL STS endpoint, since the
// global endpoint does not serve that operation. A JWT that carries a
// `jti` is accepted by Anthropic once, so the file is rewritten with a
// fresh token well before the previous one's lifetime ends.
//
// Environment (all optional but the region):
//
//	AWS_REGION / AWS_DEFAULT_REGION            STS region (required)
//	ANTHROPIC_IDENTITY_TOKEN_FILE              default /var/run/secrets/anthropic.com/token
//	ANTHROPIC_IDENTITY_TOKEN_AUDIENCE          default https://api.anthropic.com
//	ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS  token lifetime, 60-3600, default 600
//	ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS   rewrite interval, default 240, < duration
//	ANTHROPIC_IDENTITY_TOKEN_SIGNING_ALGORITHM RS256 (default) or ES384
//
// The token is never logged. Exit status: 0 on success (-once) or clean
// shutdown (loop), 1 on a failed -once, 2 on bad configuration.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	once := flag.Bool("once", false, "mint one token, write it and exit")
	flag.Parse()

	logger := log.New(os.Stderr, "[id-token] ", log.LstdFlags|log.LUTC)
	cfg, err := configFromEnv(os.Getenv)
	if err != nil {
		logger.Printf("ERROR: %v", err)
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	src, err := newSTSSource(ctx, cfg)
	if err != nil {
		logger.Printf("ERROR: %v", err)
		os.Exit(1)
	}

	if *once {
		if err := mintOnce(ctx, cfg, src, logger); err != nil {
			logger.Printf("ERROR: %v", err)
			os.Exit(1)
		}
		return
	}
	if err := loop(ctx, cfg, src, logger, sleepCtx); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintf(os.Stderr, "[id-token] loop ended: %v\n", err)
		os.Exit(1)
	}
}
