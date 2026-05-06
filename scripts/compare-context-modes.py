#!/usr/bin/env python3
"""
Compare "full" vs "compiled" contextMode on the sphinx-bounties repo.

Runs 3 turns under each mode (separate sessions) sequentially,
polls until each turn completes, then prints a side-by-side token table,
quality metrics, LLM scores, and saves full results to scripts/results/.

Usage:
  python3 scripts/compare-context-modes.py

Optional env vars:
  SERVER=http://localhost:3355   (default)
  REPO_URL=https://github.com/stakwork/sphinx-bounties  (default)
  POLL_INTERVAL=10               seconds between status polls (default)
  POLL_MAX=60                    max polls per turn before giving up (default)
  ANTHROPIC_API_KEY              for LLM quality scoring (optional)
  SCORE_MODEL=claude-opus-4-5    model used for LLM scoring (default)
"""

import datetime
import json
import os
import sys
import time
import urllib.request
import urllib.error
from uuid import uuid4

SERVER       = os.environ.get("SERVER", "http://localhost:3355")
REPO_URL     = os.environ.get("REPO_URL", "https://github.com/stakwork/sphinx-bounties")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
POLL_MAX      = int(os.environ.get("POLL_MAX", "60"))

TURNS = [
    "How does authentication work? Walk me through the challenge/verify flow and the key functions involved.",
    "When a bounty payment is made, what happens end-to-end? Show me the key functions and how they connect.",
    "What are all the admin endpoints and what do they do? Then explain in detail how the reconcile endpoint works.",
]

# ─── helpers ─────────────────────────────────────────────────────────────────

def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)

def poll_until_done(request_id: str, label: str) -> dict:
    url = f"{SERVER}/progress?request_id={request_id}"
    for i in range(1, POLL_MAX + 1):
        time.sleep(POLL_INTERVAL)
        try:
            data = get_json(url)
        except Exception as e:
            print(f"    [{label}] poll {i}: error {e}", flush=True)
            continue
        status = data.get("status", "")
        print(f"    [{label}] poll {i}: {status}", flush=True)
        if status in ("completed", "failed"):
            return data
    return {"status": "timed_out"}

def extract_usage(result: dict) -> dict:
    u = result.get("result", {}).get("usage", {}) or {}
    agent = u.get("agent", {}) or {}
    context_summary = u.get("contextSummary", {}) or {}
    return {
        "prompt":     u.get("inputTokens", u.get("promptTokens", u.get("prompt_tokens", 0))),
        "completion": u.get("outputTokens", u.get("completionTokens", u.get("completion_tokens", 0))),
        "total":      u.get("totalTokens", u.get("total_tokens", 0)),
        "agent_total": agent.get("totalTokens", 0),
        "summary_total": context_summary.get("totalTokens", 0),
    }

def run_session(mode: str) -> list[dict]:
    """Run all 3 turns under one session with the given contextMode."""
    session_id = str(uuid4())
    print(f"\n{'─'*60}", flush=True)
    print(f"MODE: {mode.upper()}   session={session_id}", flush=True)
    print(f"{'─'*60}", flush=True)

    turns = []
    for i, prompt in enumerate(TURNS, 1):
        print(f"\n  Turn {i}: {prompt[:70]}...", flush=True)
        try:
            resp = post_json(f"{SERVER}/repo/agent", {
                "repo_url":    REPO_URL,
                "prompt":      prompt,
                "sessionId":   session_id,
                "contextMode": mode,
            })
        except Exception as e:
            print(f"    [error sending turn {i}]: {e}", flush=True)
            turns.append({"turn": i, "status": "send_error", "usage": {}, "answer": ""})
            continue

        request_id = resp.get("request_id", "")
        if not request_id:
            print(f"    [no request_id in response]: {resp}", flush=True)
            turns.append({"turn": i, "status": "no_request_id", "usage": {}, "answer": ""})
            continue

        result = poll_until_done(request_id, f"{mode} T{i}")
        usage  = extract_usage(result)
        answer = (result.get("result", {}) or {}).get("final_answer", "") or ""
        turns.append({
            "turn":   i,
            "status": result.get("status", "?"),
            "usage":  usage,
            "answer": answer,
        })

    return turns

# ─── main ────────────────────────────────────────────────────────────────────

def main():
    full_turns     = run_session("full")
    compiled_turns = run_session("compiled")

    # ── token table ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("TOKEN COMPARISON")
    print(f"{'═'*72}")
    header = f"{'Turn':<6} {'Mode':<10} {'Prompt':>8} {'Compl':>8} {'Agent':>8} {'Summary':>8} {'Total':>8}   Status"
    print(header)
    print("─" * 72)

    for t in full_turns + compiled_turns:
        u    = t["usage"]
        mode = "full" if t in full_turns else "compiled"
        print(f"{t['turn']:<6} {mode:<10} {u.get('prompt',0):>8} {u.get('completion',0):>8} {u.get('agent_total',0):>8} {u.get('summary_total',0):>8} {u.get('total',0):>8}   {t['status']}")

    # ── delta summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("TURN-BY-TURN DELTA  (compiled - full, negative = compiled used fewer tokens)")
    print(f"{'═'*72}")
    print(f"{'Turn':<6} {'Prompt Δ':>10} {'Compl Δ':>10} {'Agent Δ':>10} {'Summary Δ':>10} {'Total Δ':>10}")
    print("─" * 72)
    for i in range(len(TURNS)):
        fu = full_turns[i]["usage"]     if i < len(full_turns)     else {}
        cu = compiled_turns[i]["usage"] if i < len(compiled_turns) else {}
        dp = cu.get("prompt",0)     - fu.get("prompt",0)
        dc = cu.get("completion",0) - fu.get("completion",0)
        da = cu.get("agent_total",0) - fu.get("agent_total",0)
        ds = cu.get("summary_total",0) - fu.get("summary_total",0)
        dt = cu.get("total",0)      - fu.get("total",0)
        sign = lambda x: f"+{x}" if x > 0 else str(x)
        print(f"{i+1:<6} {sign(dp):>10} {sign(dc):>10} {sign(da):>10} {sign(ds):>10} {sign(dt):>10}")

    # ── answer quality ────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("ANSWER SNIPPETS (first 400 chars each)")
    print(f"{'═'*72}")
    for i in range(len(TURNS)):
        print(f"\n--- Turn {i+1} ---")
        print(f"Q: {TURNS[i]}")
        for label, turns in [("FULL", full_turns), ("COMPILED", compiled_turns)]:
            ans = turns[i]["answer"] if i < len(turns) else "(missing)"
            print(f"\n  [{label}]\n  {ans[:400]}")

    print(f"\n{'═'*72}")
    print("Done.")

if __name__ == "__main__":
    main()
