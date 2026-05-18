package auth

// Test helpers shared across the auth package's unit tests.
//
// The strategy: we build real signed macaroons against a real (in-
// memory) trust registry and a real Redis (miniredis), and exercise
// the adapter through its public entry points. No mock fakes —
// brittleness from mock drift is exactly what the phase-4 doc warned
// against ("fixtures are the contract"). For the trust registry
// specifically we use trust.New() directly because the public
// surface is small enough that we don't gain anything from a fake.

import (
	"encoding/hex"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/decred/dcrd/dcrec/secp256k1/v4"
	"github.com/redis/go-redis/v9"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

// Deterministic test keys. Reused across every test in this package
// so a failure's diagnostic strings stay comparable across runs.
const (
	testOrgPrivHex  = "0000000000000000000000000000000000000000000000000000000000000001"
	testUserPrivHex = "1111111111111111111111111111111111111111111111111111111111111111"
	testOrgID       = "org_test"
	testUserID      = "u_alice"
)

func testOrgPriv() []byte  { return mustHexBytes(testOrgPrivHex) }
func testUserPriv() []byte { return mustHexBytes(testUserPrivHex) }

func testOrgPubHex() string {
	p := secp256k1.PrivKeyFromBytes(testOrgPriv())
	return hex.EncodeToString(p.PubKey().SerializeCompressed())
}

func mustHexBytes(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

// newTestRegistry builds an in-memory trust registry populated with
// the deterministic test org. Helper because every test that hits
// the verifier needs this setup.
func newTestRegistry(t *testing.T) *trust.Registry {
	t.Helper()
	r := trust.New()
	o := trust.Org{
		OrgID:                 testOrgID,
		Pubkey:                testOrgPubHex(),
		IssuerURL:             "https://hive.test",
		RevocationPollSeconds: 60,
	}
	if err := r.Upsert(o, trust.SeedSourceAPI); err != nil {
		t.Fatalf("seed registry: %v", err)
	}
	return r
}

// newMiniRedis spins up miniredis and points redisclient at it.
// Returns the miniredis handle for direct state inspection. The
// cleanup hook closes the miniredis AND nils out the package
// client so the next test starts clean.
func newMiniRedis(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	mr := miniredis.RunT(t)
	c := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	redisclient.SetClientForTest(c)
	t.Cleanup(func() {
		redisclient.SetClientForTest(nil)
		_ = c.Close()
	})
	return mr
}

// buildMacaroon assembles a fully-signed wire-format macaroon
// against the test keys. Knobs let each test tweak the fields it
// cares about; everything else is fixed and minimal.
type macaroonOptions struct {
	orgID      string
	userID     string
	realms     []string
	agents     []string
	invRealm   string
	invAgents  []string
	runID      string
	maxCostUSD float64
	maxSteps   int
	uaExp      time.Time
	uaIAT      time.Time
	invExp     time.Time
	invIAT     time.Time
	uaNonce    string
	invNonce   string
	budget     *macaroon.UserBudget
	// signWithPriv overrides the org private key — used in tests
	// that need to exercise the "signed by an untrusted key" path.
	signWithOrgPriv []byte
}

func defaultMacaroonOptions(now time.Time) macaroonOptions {
	return macaroonOptions{
		orgID:      testOrgID,
		userID:     testUserID,
		realms:     []string{"w1"},
		agents:     []string{"coder"},
		invRealm:   "w1",
		invAgents:  []string{"coder"},
		runID:      "r_test_run_000000000000000",
		maxCostUSD: 5.00,
		maxSteps:   100,
		uaIAT:      now.Add(-1 * time.Hour),
		uaExp:      now.Add(24 * time.Hour),
		invIAT:     now.Add(-1 * time.Minute),
		invExp:     now.Add(10 * time.Minute),
		uaNonce:    "aaaa000000000000000000000000aaaa",
		invNonce:   "bbbb000000000000000000000000bbbb",
	}
}

func buildMacaroon(t *testing.T, opts macaroonOptions) string {
	t.Helper()

	userPub, err := macaroon.Ed25519PublicKey(testUserPriv())
	if err != nil {
		t.Fatalf("derive user pubkey: %v", err)
	}

	ua := macaroon.UserAuthorization{
		UserID:     opts.userID,
		UserPubkey: macaroon.PubKey{Alg: macaroon.AlgEd25519, Key: macaroon.BytesToHex(userPub)},
		Permissions: macaroon.UserPermissions{
			Realms: opts.realms,
			Agents: opts.agents,
		},
		Budget: opts.budget,
		IAT:    opts.uaIAT.UTC().Format(time.RFC3339),
		Exp:    opts.uaExp.UTC().Format(time.RFC3339),
		Nonce:  opts.uaNonce,
	}

	signKey := opts.signWithOrgPriv
	if signKey == nil {
		signKey = testOrgPriv()
	}
	signedUA, err := macaroon.SignUserAuthorizationSingle(ua, signKey)
	if err != nil {
		t.Fatalf("sign ua: %v", err)
	}

	inv := macaroon.Invocation{
		Realm:      opts.invRealm,
		Agents:     opts.invAgents,
		RunID:      opts.runID,
		MaxCostUSD: opts.maxCostUSD,
		MaxSteps:   opts.maxSteps,
		IAT:        opts.invIAT.UTC().Format(time.RFC3339),
		Exp:        opts.invExp.UTC().Format(time.RFC3339),
		Nonce:      opts.invNonce,
	}
	signedInv, err := macaroon.SignInvocation(inv, testUserPriv())
	if err != nil {
		t.Fatalf("sign invocation: %v", err)
	}

	m := macaroon.Macaroon{
		V:                 1,
		OrgID:             opts.orgID,
		UserAuthorization: signedUA,
		Invocation:        signedInv,
		Attenuations:      []macaroon.Attenuation{},
	}
	encoded, err := macaroon.EncodeMacaroon(m)
	if err != nil {
		t.Fatalf("encode macaroon: %v", err)
	}
	return encoded
}
