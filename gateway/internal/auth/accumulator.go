package auth

import (
	"context"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/duration"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Redis key sub-paths for the phase-6 accumulators (the bifrost:
// prefix is added by redisclient.Key). Shapes per
// gateway/plans/phases/phase-6-plugin-enforcement.md "Redis schema".
const (
	costRunPrefix   = "cost:run:"   // HASH { total: float }
	stepsRunPrefix  = "steps:run:"  // HASH { total: int }
	toolsRunPrefix  = "tools:run:"  // LIST, capped at toolHistoryLen
	costUAPrefix    = "cost:ua:"    // HASH { total: float }
	costAgentPrefix = "cost:agent:" // HASH { total: float }, key + ":" + bucket
)

// toolHistoryLen is the tool-loop detection window: tools:run keeps
// the last N tool names (LPUSH + LTRIM 0 N-1).
const toolHistoryLen = 10

// ApplyToLLMPost is the canonical "call from PostLLMHook" entry
// point for the phase-6 accumulator writes. It walks the verified
// chain and issues one pipelined Redis round-trip:
//
//   - HINCRBYFLOAT cost:run:<r> / HINCRBY steps:run:<r> + EXPIRE,
//     for every distinct run_id in the chain (leaf + ancestors) —
//     killing or capping a parent must see descendant spend.
//   - HINCRBYFLOAT cost:ua:<ua.nonce> + EXPIRE, only when the UA
//     carried a cumulative budget (phase-4 "budget envelope").
//   - HINCRBYFLOAT cost:agent:<agent>:<bucket> + EXPIRE, only when
//     the leaf agent has a configured windowed budget.
//   - LPUSH/LTRIM tools:run:<leaf> + EXPIRE when the response
//     contained tool calls.
//
// Fire-and-forget per the phase-6 failure-mode contract: accounting
// fails open. The pipeline runs on a goroutine with its own timeout;
// errors log loudly and are never surfaced to the caller — a Redis
// outage must not block or fail the response. The resulting drift is
// bounded by outage duration × call rate and is reconciled against
// logs.db (which Bifrost's own logging plugin writes after us).
//
// No-op when claims is nil (shadow mode without a verified macaroon)
// or Redis is unconfigured (observability mode).
func ApplyToLLMPost(claims *macaroon.Claims, costUSD float64, toolNames []string) {
	if claims == nil || redisclient.Client() == nil {
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), pipelineTimeout)
		defer cancel()
		if err := accumulate(ctx, claims, costUSD, toolNames, time.Now().UTC()); err != nil {
			pluginlog.Warnf(
				"auth: accumulator pipeline failed (spend uncounted this call) run_id=%s agent=%s cost=%.4f: %v",
				claims.RunID, claims.AgentName, costUSD, err,
			)
		}
	}()
}

// accumulate is the synchronous pipeline body. Split from
// ApplyToLLMPost so tests can run it deterministically against
// miniredis without racing the goroutine.
//
// Every command is individually idempotent on its own value
// (HINCRBYFLOAT is commutative, LPUSH+LTRIM is bounded, EXPIRE is
// set-not-add), so the pipeline needs no transaction — out-of-order
// or partially-applied writes converge to the same state. See
// phase-6 "Pipelining and atomicity".
func accumulate(
	ctx context.Context,
	claims *macaroon.Claims,
	costUSD float64,
	toolNames []string,
	now time.Time,
) error {
	rdb := redisclient.Client()
	if rdb == nil {
		return nil
	}
	pipe := rdb.Pipeline()

	// Per-run accumulators: every distinct run_id in the chain,
	// outermost first. Each layer's keys get that layer's own TTL
	// axis (clamp(layer.exp - now + 1h, 1h, 7d)) — an ancestor's
	// accumulator must outlive the short-lived leaf that wrote it.
	// TTL refreshes on every write, so an actively-spending run
	// keeps its keys alive for its whole lifetime.
	seen := make(map[string]bool, len(claims.Chain))
	leafTTL := runKeyTTL(parseRFC3339(claims.EffectiveCaveats.Exp), now)
	for _, layer := range claims.Chain {
		if layer.RunID == "" || seen[layer.RunID] {
			continue
		}
		seen[layer.RunID] = true
		ttl := runKeyTTL(parseRFC3339(layer.Exp), now)
		costKey := redisclient.Key(costRunPrefix + layer.RunID)
		stepsKey := redisclient.Key(stepsRunPrefix + layer.RunID)
		pipe.HIncrByFloat(ctx, costKey, "total", costUSD)
		pipe.HIncrBy(ctx, stepsKey, "total", 1)
		pipe.Expire(ctx, costKey, ttl)
		pipe.Expire(ctx, stepsKey, ttl)
	}

	// UA cumulative envelope — only when the org actually set one.
	// No bucket ⇒ no enforcement; per-invocation caps are checked at
	// signature time and need no Redis state.
	if claims.UABudget != nil && claims.UABudget.MaxTotalUSD > 0 && claims.UANonce != "" {
		uaKey := redisclient.Key(costUAPrefix + claims.UANonce)
		uaTTL := runKeyTTL(parseRFC3339(claims.UAExp), now)
		pipe.HIncrByFloat(ctx, uaKey, "total", costUSD)
		pipe.Expire(ctx, uaKey, uaTTL)
	}

	// Per-agent windowed bucket — only when the operator configured
	// a budget for this agent. The bucket key is computed from THIS
	// write's clock ("bucket by the time of the write" — see phase-6
	// "Bucket boundary mid-call"), never carried over from PreHook.
	if b, ok := GetConfig().AgentBudgets[claims.AgentName]; ok && b.CapUSD > 0 && b.Window != "" {
		if w, err := duration.Parse(b.Window); err != nil {
			pluginlog.Warnf("auth: accumulator agent=%s: unrecognized window %q", claims.AgentName, b.Window)
		} else {
			agentKey := redisclient.Key(costAgentPrefix + claims.AgentName + ":" + w.BucketKey(now))
			pipe.HIncrByFloat(ctx, agentKey, "total", costUSD)
			pipe.Expire(ctx, agentKey, w.TTL())
		}
	}

	// Tool-loop history on the leaf run only.
	if len(toolNames) > 0 && claims.RunID != "" {
		toolsKey := redisclient.Key(toolsRunPrefix + claims.RunID)
		for _, name := range toolNames {
			pipe.LPush(ctx, toolsKey, name)
		}
		pipe.LTrim(ctx, toolsKey, 0, toolHistoryLen-1)
		pipe.Expire(ctx, toolsKey, leafTTL)
	}

	_, err := pipe.Exec(ctx)
	return err
}
