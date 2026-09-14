package auth

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/duration"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// The cap walk — phase-6 "Hot path" PIPELINE 2. Reads the
// accumulators PostLLMHook writes (accumulator.go) and compares them
// against every cap the verified macaroon and the plugin config
// declare. One pipelined Redis round-trip, then a fixed-order
// evaluation so the rejection reason is deterministic:
//
//  1. cost:run:<r>   >= layer.MaxCostUSD   for every chain layer  → run_cost_exceeded
//  2. cost:ua:<n>    >= realm cap          (phase 11)             → realm_budget_exceeded
//  3. cost:ua:<n>    >= ua.MaxTotalUSD                            → ua_budget_exceeded
//  4. steps:run:<r>  >= layer.MaxSteps     for every chain layer  → run_step_exceeded
//  5. cost:agent:<a>:<bucket> >= agent_budgets cap                → agent_cost_exceeded
//
// The walk covers every layer, leaf first, then each ancestor: a
// child whose own $2 cap is untouched is still rejected when its
// parent's $5 is gone, because the parent's counter includes every
// descendant's spend. Steps are walked the same way (the plan only
// names the leaf; walking ancestors is strictly stronger and matches
// what the accumulator increments).
//
// Comparisons use *past* spend — the call's own price isn't known
// before it runs — so a run can overshoot by up to one call's cost
// (times concurrency). Phase 6 "Pipelining and atomicity" accepts
// that. `>=` rather than `>` so a run that lands exactly on its cap
// is done.
//
// A cap of 0 means "no cap" for macaroon caveats (the field is
// optional at every layer). For agent_budgets it means the opposite:
// an entry with cap_usd 0 blocks the agent outright — that is how
// phase 6 says to stop an agent permanently (a kill:agent key only
// lasts 24h).
//
// Fail-closed on Redis errors (402 budget_check_unavailable); no-op
// in observability mode (Redis unconfigured), like CheckRevocations.

// capLayer is one budgeted run in the walk order.
type capLayer struct {
	runID      string
	maxCostUSD float64
	maxSteps   int
}

// CheckCaps runs the cap walk. realmID is the swarm's own realm (from
// the trust registry; "" for single-swarm deployments), used to pick
// the per-realm cap out of the macaroon's realm_budgets. Returns nil
// on "within every cap".
func CheckCaps(ctx context.Context, claims *macaroon.Claims, realmID string, now time.Time) *AdapterError {
	rdb := redisclient.Client()
	if rdb == nil {
		return nil
	}
	if claims == nil {
		return &AdapterError{
			Code:       "verify_internal_error",
			HTTPStatus: 401,
			Message:    "nil claims passed to cap walk",
		}
	}

	layers := capLayers(claims)
	realmCap := realmCapFor(claims, realmID)
	var uaCap float64
	if claims.UABudget != nil {
		uaCap = claims.UABudget.MaxTotalUSD
	}
	agentBudget, agentConfigured := GetConfig().AgentBudgets[claims.AgentName]

	pctx, cancel := context.WithTimeout(ctx, pipelineTimeout)
	defer cancel()
	pipe := rdb.Pipeline()

	costCmds := make([]*redis.StringCmd, len(layers))
	stepCmds := make([]*redis.StringCmd, len(layers))
	for i, l := range layers {
		if l.maxCostUSD > 0 {
			costCmds[i] = pipe.HGet(pctx, redisclient.Key(costRunPrefix+l.runID), "total")
		}
		if l.maxSteps > 0 {
			stepCmds[i] = pipe.HGet(pctx, redisclient.Key(stepsRunPrefix+l.runID), "total")
		}
	}
	var uaCmd *redis.StringCmd
	if claims.UANonce != "" && (uaCap > 0 || realmCap > 0) {
		uaCmd = pipe.HGet(pctx, redisclient.Key(costUAPrefix+claims.UANonce), "total")
	}
	var agentCmd *redis.StringCmd
	var agentBucket string
	if agentConfigured && agentBudget.CapUSD > 0 && agentBudget.Window != "" {
		if w, err := duration.Parse(agentBudget.Window); err == nil {
			agentBucket = w.BucketKey(now)
			agentCmd = pipe.HGet(pctx, redisclient.Key(costAgentPrefix+claims.AgentName+":"+agentBucket), "total")
		}
		// An unparseable window is logged by the accumulator on
		// every write; nothing to compare here, so no cap applies.
	}

	if _, err := pipe.Exec(pctx); err != nil && !errors.Is(err, redis.Nil) {
		return budgetUnavailable(fmt.Sprintf("redis pipeline: %v", err))
	}

	// 1. Per-run cost, leaf first.
	for i, l := range layers {
		if costCmds[i] == nil {
			continue
		}
		spent, err := hashFloat(costCmds[i])
		if err != nil {
			return budgetUnavailable(fmt.Sprintf("cost:run:%s: %v", l.runID, err))
		}
		if spent >= l.maxCostUSD {
			return &AdapterError{
				Code:       "run_cost_exceeded",
				HTTPStatus: 402,
				Message:    fmt.Sprintf("run %s spent $%.4f of its $%.2f cap", l.runID, spent, l.maxCostUSD),
			}
		}
	}

	// 2-3. UA envelope: realm cap (phase 11) then the org-wide total.
	if uaCmd != nil {
		spent, err := hashFloat(uaCmd)
		if err != nil {
			return budgetUnavailable(fmt.Sprintf("cost:ua: %v", err))
		}
		if realmCap > 0 && spent >= realmCap {
			return &AdapterError{
				Code:       "realm_budget_exceeded",
				HTTPStatus: 402,
				Message: fmt.Sprintf("user %s spent $%.4f of the $%.2f cap for realm %s",
					claims.UserID, spent, realmCap, realmID),
			}
		}
		if uaCap > 0 && spent >= uaCap {
			return &AdapterError{
				Code:       "ua_budget_exceeded",
				HTTPStatus: 402,
				Message: fmt.Sprintf("user %s spent $%.4f of the $%.2f authorization envelope",
					claims.UserID, spent, uaCap),
			}
		}
	}

	// 4. Per-run steps, same walk.
	for i, l := range layers {
		if stepCmds[i] == nil {
			continue
		}
		steps, err := hashInt(stepCmds[i])
		if err != nil {
			return budgetUnavailable(fmt.Sprintf("steps:run:%s: %v", l.runID, err))
		}
		if steps >= int64(l.maxSteps) {
			return &AdapterError{
				Code:       "run_step_exceeded",
				HTTPStatus: 402,
				Message:    fmt.Sprintf("run %s used %d of its %d steps", l.runID, steps, l.maxSteps),
			}
		}
	}

	// 5. Per-agent windowed budget.
	if agentConfigured && agentBudget.CapUSD <= 0 {
		return &AdapterError{
			Code:       "agent_cost_exceeded",
			HTTPStatus: 402,
			Message:    fmt.Sprintf("agent %s has a $0 budget (blocked by operator config)", claims.AgentName),
		}
	}
	if agentCmd != nil {
		spent, err := hashFloat(agentCmd)
		if err != nil {
			return budgetUnavailable(fmt.Sprintf("cost:agent:%s: %v", claims.AgentName, err))
		}
		if spent >= agentBudget.CapUSD {
			return &AdapterError{
				Code:       "agent_cost_exceeded",
				HTTPStatus: 402,
				Message: fmt.Sprintf("agent %s spent $%.4f of its $%.2f/%s cap (bucket %s)",
					claims.AgentName, spent, agentBudget.CapUSD, agentBudget.Window, agentBucket),
			}
		}
	}

	return nil
}

// capLayers returns the chain's budgeted runs in walk order: leaf
// first, then each ancestor outward. Claims.Chain is outermost-first;
// when it's empty (older callers, tests) the leaf is synthesized from
// Claims.RunID + EffectiveCaveats.
func capLayers(claims *macaroon.Claims) []capLayer {
	if len(claims.Chain) == 0 {
		if claims.RunID == "" {
			return nil
		}
		return []capLayer{{
			runID:      claims.RunID,
			maxCostUSD: claims.EffectiveCaveats.MaxCostUSD,
			maxSteps:   claims.EffectiveCaveats.MaxSteps,
		}}
	}
	out := make([]capLayer, 0, len(claims.Chain))
	seen := make(map[string]bool, len(claims.Chain))
	for i := len(claims.Chain) - 1; i >= 0; i-- {
		l := claims.Chain[i]
		if l.RunID == "" || seen[l.RunID] {
			continue
		}
		seen[l.RunID] = true
		out = append(out, capLayer{runID: l.RunID, maxCostUSD: l.MaxCostUSD, maxSteps: l.MaxSteps})
	}
	return out
}

// realmCapFor returns the narrowed per-realm cap for this swarm's
// realm, or 0 when the macaroon carries no realm_budgets, the swarm
// has no realm_id, or the realm isn't listed (membership is checked
// separately by CheckRealmMembership; here absent just means no cap).
func realmCapFor(claims *macaroon.Claims, realmID string) float64 {
	if realmID == "" || claims.EffectiveCaveats.Budget == nil {
		return 0
	}
	return claims.EffectiveCaveats.Budget.RealmBudgets[realmID].MaxTotalUSD
}

func budgetUnavailable(detail string) *AdapterError {
	return &AdapterError{
		Code:       "budget_check_unavailable",
		HTTPStatus: 402,
		Message:    detail,
	}
}

// hashFloat / hashInt read an HGET result, treating a missing key or
// field (redis.Nil) as zero — a run that has never spent is at $0.
func hashFloat(cmd *redis.StringCmd) (float64, error) {
	v, err := cmd.Float64()
	if errors.Is(err, redis.Nil) {
		return 0, nil
	}
	return v, err
}

func hashInt(cmd *redis.StringCmd) (int64, error) {
	v, err := cmd.Int64()
	if errors.Is(err, redis.Nil) {
		return 0, nil
	}
	return v, err
}
