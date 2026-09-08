#!/usr/bin/env python3
"""Generate the 'everything goes in the graph' diagram series.

Three tiers of increasing size, a few variants each:
  concepts  (3-4 nodes)
  code      (7-10 nodes)
  trace     (>12 nodes)
Each output is a self-contained HTML using system-canvas-standalone from unpkg.
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

PRELUDE = r"""
      const HUE = {
        repository: '#f59e0b', concept: '#34d399', file: '#38bdf8', function: '#a78bfa',
        class: '#e879f9', endpoint: '#22d3ee', datamodel: '#fb7185',
        pr: '#94a3b8', commit: '#94a3b8', clue: '#fde68a', proposal: '#fdba74',
        chat: '#f472b6', run: '#fb923c', session: '#fbbf24', toolcall: '#cbd5e1', workflow: '#a3e635',
        note: '#64748b',
      };
      const FRESH = { fresh: '#2fa57c', aging: '#fbbf24', stale: '#475569' };
      const cd = (c) => c.node.customData || {};
      const tint = (hex, a) => { const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16); return `rgba(${r},${g},${b},${a})`; };
      const statusColor = (c) => FRESH[cd(c).freshness === 'growing' ? 'fresh' : (cd(c).freshness || 'aging')];
      const LABEL = { fresh: 'FRESH', aging: 'AGING', stale: 'STALE', growing: 'GROWING' };
      const cat = (key, label, w, h, extra = {}) => {
        const { slots = {}, ...rest } = extra;
        return { defaultWidth: w, defaultHeight: h, cornerRadius: 10, fill: tint(HUE[key], 0.15), stroke: HUE[key],
          slots: { header: { kind: 'text', value: label }, ...slots }, ...rest };
      };
      const categories = {
        repository: cat('repository', ':Repository', 240, 96, { cornerRadius: 14, slots: {
          footer: { kind: 'text', value: (c) => cd(c).footer || '' } } }),
        concept: cat('concept', ':Concept', 200, 84, { slots: {
          topRight: { kind: 'pill', value: (c) => LABEL[cd(c).freshness] || '', color: statusColor },
          topRightOuter: { kind: 'count', value: (c) => cd(c).clues, hideWhenEmpty: true },
          footer: { kind: 'text', value: (c) => cd(c).footer || '' } } }),
        file: cat('file', ':File', 250, 72, { slots: {
          topRight: { kind: 'pill', value: (c) => cd(c).inDocs === undefined ? '' : (cd(c).inDocs ? 'IN DOCS' : 'OFF DOCS'), color: (c) => cd(c).inDocs ? HUE.concept : HUE.note },
          bottomEdge: { kind: 'progress', value: (c) => cd(c).importance || 0, color: statusColor, hideWhenZero: true } } }),
        function: cat('function', ':Function', 180, 64, { slots: { topRightOuter: { kind: 'count', value: (c) => cd(c).callers, hideWhenEmpty: true } } }),
        class: cat('class', ':Class', 180, 64),
        endpoint: cat('endpoint', ':Endpoint', 200, 64),
        datamodel: cat('datamodel', ':Datamodel', 180, 64, { cornerRadius: 32 }),
        pr: cat('pr', ':PullRequest', 230, 76, { slots: { topRight: { kind: 'pill', value: (c) => cd(c).when || '', color: statusColor } } }),
        commit: cat('commit', ':Commit', 230, 76, { slots: { topRight: { kind: 'pill', value: (c) => cd(c).when || '', color: statusColor } } }),
        clue: cat('clue', ':Clue', 200, 72),
        proposal: cat('proposal', ':ConceptProposal', 200, 72, { slots: { topRight: { kind: 'pill', value: (c) => cd(c).status || '', color: HUE.proposal } } }),
        chat: cat('chat', ':StrutChat', 200, 72),
        run: cat('run', ':StrutRun', 200, 84, { slots: {
          topRight: { kind: 'pill', value: (c) => cd(c).status || '', color: (c) => cd(c).status === 'OK' ? FRESH.fresh : HUE.run },
          footer: { kind: 'text', value: (c) => cd(c).footer || '' } } }),
        session: cat('session', ':StrutAgentSession', 210, 84, { slots: { footer: { kind: 'text', value: (c) => cd(c).footer || '' } } }),
        toolcall: cat('toolcall', ':StrutToolCall', 190, 64, { slots: { topRightOuter: { kind: 'count', value: (c) => cd(c).n, hideWhenEmpty: true } } }),
        workflow: cat('workflow', ':StrutWorkflow', 200, 72, { slots: { topRight: { kind: 'pill', value: (c) => cd(c).version || '', color: HUE.workflow } } }),
        note: { defaultWidth: 260, defaultHeight: 56, cornerRadius: 8, fill: 'rgba(15,23,42,0.55)', stroke: HUE.note },
      };
      const N = (id, category, text, x, y, customData, extra = {}) => ({ id, type: 'text', text, x, y, category, customData, ...extra });
      const E = (from, to, label, opts = {}) => ({ id: `${from}->${to}:${label || ''}`, fromNode: from, toNode: to, label, ...opts });
      const theme = { ...SystemCanvas.themes.midnight, categories: { ...SystemCanvas.themes.midnight.categories, ...categories } };
"""

TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      html, body {{ margin: 0; height: 100%; background: #0a0a0a; }}
      #app {{ width: 100vw; height: 100vh; }}
    </style>
  </head>
  <body>
    <div id="app"></div>
    <script src="https://unpkg.com/system-canvas-standalone@latest/dist/system-canvas.min.js"></script>
    <script>
      // {subtitle}
{prelude}
      const canvas = {{
        theme: {{ base: 'midnight' }},
        nodes: [
{nodes}
        ],
        edges: [
{edges}
        ],
      }};
      SystemCanvas.render(document.getElementById('app'), {{ canvas, theme, edgeStyle: 'bezier', rootLabel: {root_label} }});
    </script>
  </body>
</html>
"""

def js(v):
    return json.dumps(v, ensure_ascii=False)

def node(id, category, text, x, y, data=None, **extra):
    parts = [js(id), js(category), js(text), str(x), str(y), js(data or {})]
    if extra:
        parts.append(js(extra))
    return f"          N({', '.join(parts)}),"

SIDES = {"down": ("bottom", "top"), "up": ("top", "bottom"), "right": ("right", "left"), "left": ("left", "right"), "auto": None}

def edge(f, t, label=None, sides="down", **opts):
    if SIDES.get(sides):
        opts.setdefault("fromSide", SIDES[sides][0]); opts.setdefault("toSide", SIDES[sides][1])
    s = f"          E({js(f)}, {js(t)}, {js(label) if label else 'undefined'}"
    if opts:
        s += f", {js(opts)}"
    return s + "),"

def w(imp):  # edge width from importance
    return round(1.5 + imp * 6.5, 1)

G, A, S = "#2fa57c", "#fbbf24", "#475569"  # fresh / aging / stale edge colors

DIAGRAMS = []

def add(slug, title, subtitle, nodes, edges, group=None):
    DIAGRAMS.append((slug, title, subtitle, nodes, edges))

# ---------------------------------------------------------------- Tier 1: concepts
add("01-concepts-hierarchy", "Concepts: hierarchy", "A repo owns concepts; concepts nest (PARENT_OF). 4 nodes.",
    [node("repo", "repository", "stakwork/hive", -110, -160, {"footer": "3 concepts"}),
     node("ws", "concept", "Workspaces", -320, 0, {"freshness": "fresh", "clues": 23, "footer": "created 2025-11-02"}),
     node("runner", "concept", "Task Runner", 100, 0, {"freshness": "fresh", "clues": 17, "footer": "created 2026-01-20"}),
     node("members", "concept", "Workspace Members", -320, 170, {"freshness": "growing", "clues": 6, "footer": "created 2026-06-14"})],
    [edge("repo", "ws", "importance 0.95", color=G, strokeWidth=w(0.95)),
     edge("repo", "runner", "importance 0.88", color=G, strokeWidth=w(0.88)),
     edge("ws", "members", "PARENT_OF", color=G, strokeWidth=4)])

add("02-concepts-provenance", "Concepts: what grows one", "A PR and a commit TOUCH a concept; the concept MODIFIES a file. 4 nodes.",
    [node("pr", "pr", "PR #412\nAdd workspace invites", -420, -90, {"when": "2D AGO", "freshness": "fresh"}),
     node("sha", "commit", "a1b2c3f\nworkspace: slug rules", -420, 90, {"when": "5H AGO", "freshness": "fresh"}),
     node("ws", "concept", "Workspaces", -60, 0, {"freshness": "fresh", "clues": 23, "footer": "prNumbers · commitShas · docs"}),
     node("f", "file", "src/services/workspace.ts", 300, 0, {"inDocs": True, "importance": 0.92, "freshness": "fresh"})],
    [edge("pr", "ws", "TOUCHES", sides="right", color=G, strokeWidth=2.5),
     edge("sha", "ws", "TOUCHES", sides="right", color=G, strokeWidth=2.5),
     edge("ws", "f", "MODIFIES 0.92", sides="right", color=G, strokeWidth=w(0.92))])

add("03-concepts-freshness", "Concepts: scores on the edge", "Same repo, three concepts. Edge weight = importance, color = freshness. 4 nodes.",
    [node("repo", "repository", "stakwork/hive", -110, -170, {"footer": "importance · freshness · cluesCount"}),
     node("ws", "concept", "Workspaces", -420, 20, {"freshness": "fresh", "clues": 23, "footer": "touched 2d ago"}),
     node("auth", "concept", "Auth", -100, 20, {"freshness": "aging", "clues": 11, "footer": "touched 3w ago"}),
     node("billing", "concept", "Billing", 220, 20, {"freshness": "stale", "clues": 4, "footer": "touched 9mo ago"}, color=S)],
    [edge("repo", "ws", "0.95", color=G, strokeWidth=w(0.95)),
     edge("repo", "auth", "0.72", color=A, strokeWidth=w(0.72)),
     edge("repo", "billing", "0.35", color=S, strokeWidth=w(0.35))])

add("04-concepts-everything-is-a-node", "Everything is a node", "The four kinds of thing the graph holds, and how they point at each other. 4 nodes.",
    [node("c", "concept", "Concept\npayment retry policy", -110, -180, {"freshness": "fresh", "footer": "a named idea in the domain"}, height=108),
     node("code", "function", "Code\nretryCharge()", -380, 40, {"callers": 6}, height=88),
     node("pr", "pr", "PR #4812\nswitch to fixed delay", 160, 40, {"when": "MERGED", "freshness": "fresh"}, height=88),
     node("trace", "run", "Trace\nagent run r-2291", -110, 220, {"status": "OK", "footer": "read 3 nodes · wrote 1 edge"}, height=108)],
    [edge("c", "code", "MODIFIES", sides="auto", color=G, strokeWidth=5),
     edge("pr", "c", "TOUCHES", sides="auto", color=G, strokeWidth=2.5),
     edge("pr", "code", "MODIFIES", sides="left", color="#94a3b8", strokeWidth=2),
     edge("trace", "c", "ACCESSED", sides="up", color="#fb923c", strokeWidth=2.5),
     edge("trace", "code", "ACCESSED", sides="auto", color="#fb923c", strokeWidth=2)])

# ---------------------------------------------------------------- Tier 2: code
add("05-code-vertical-slice", "Code: one concept, all the way down", "Concept → File → Function/Class/Endpoint → Datamodel. 8 nodes.",
    [node("ws", "concept", "Workspaces", -100, -200, {"freshness": "fresh", "clues": 23, "footer": "touched 2d ago"}),
     node("f1", "file", "src/services/workspace.ts", -360, -20, {"inDocs": True, "importance": 0.92, "freshness": "fresh"}),
     node("f2", "file", "src/app/api/workspaces/\nroute.ts", 140, -20, {"inDocs": True, "importance": 0.81, "freshness": "fresh"}, height=88),
     node("fn", "function", "createWorkspace()", -460, 150, {"callers": 6}),
     node("cls", "class", "WorkspaceService", -720, 150, {}),
     node("ep", "endpoint", "POST /api/workspaces", 150, 150, {}),
     node("dm", "datamodel", "Workspace", -460, 320, {}),
     node("fn2", "function", "validateSlug()", -220, 320, {"callers": 2})],
    [edge("ws", "f1", "MODIFIES 0.92", color=G, strokeWidth=w(0.92)),
     edge("ws", "f2", "MODIFIES 0.81", color=G, strokeWidth=w(0.81)),
     edge("f1", "fn", "CONTAINS", color="#a78bfa", strokeWidth=2.5),
     edge("f1", "cls", "CONTAINS", color="#e879f9", strokeWidth=2.5),
     edge("f2", "ep", "CONTAINS", color="#22d3ee", strokeWidth=2.5),
     edge("fn", "dm", "CALLS", color="#a78bfa", strokeWidth=2),
     edge("fn", "fn2", "CALLS", sides="auto", color="#a78bfa", strokeWidth=2),
     edge("ep", "fn", "CALLS", sides="left", color="#22d3ee", strokeWidth=1.5)])

add("06-code-two-branches", "Code: two branches of the tree", "Repository → two concepts → files → functions, one fresh and one stale. 9 nodes.",
    [node("repo", "repository", "stakwork/hive", -110, -260, {"footer": "2 of 5 concepts shown"}),
     node("runner", "concept", "Task Runner", -400, -80, {"freshness": "fresh", "clues": 17, "footer": "touched 5h ago"}),
     node("billing", "concept", "Billing", 200, -80, {"freshness": "stale", "clues": 4, "footer": "touched 9mo ago"}, color=S),
     node("pool", "file", "src/lib/runner/pool.ts", -560, 100, {"inDocs": True, "importance": 0.87, "freshness": "fresh"}),
     node("queue", "file", "src/lib/runner/queue.ts", -280, 100, {"inDocs": False, "importance": 0.44, "freshness": "aging"}),
     node("stripe", "file", "src/lib/stripe.ts", 190, 100, {"inDocs": False, "importance": 0.31, "freshness": "stale"}, color=S),
     node("spawn", "function", "RunnerPool.spawn()", -540, 270, {"callers": 4}),
     node("enq", "function", "enqueue()", -280, 270, {"callers": 1}),
     node("chk", "function", "createCheckout()", 200, 270, {"callers": 1}, color=S)],
    [edge("repo", "runner", "0.88", color=G, strokeWidth=w(0.88)),
     edge("repo", "billing", "0.35", color=S, strokeWidth=w(0.35)),
     edge("runner", "pool", "MODIFIES 0.87", color=G, strokeWidth=w(0.87)),
     edge("runner", "queue", "MODIFIES 0.44", color=A, strokeWidth=w(0.44)),
     edge("billing", "stripe", "MODIFIES 0.31", color=S, strokeWidth=w(0.31)),
     edge("pool", "spawn", "CONTAINS", color="#a78bfa", strokeWidth=2.5),
     edge("queue", "enq", "CONTAINS", color="#a78bfa", strokeWidth=2.5),
     edge("stripe", "chk", "CONTAINS", color=S, strokeWidth=2.5)])

add("07-code-with-provenance", "Code: a concept with its evidence", "PRs, a clue and a proposal on the left; the code it maps to on the right. 10 nodes.",
    [node("pr", "pr", "PR #412\nAdd workspace invites", -640, -140, {"when": "2D AGO", "freshness": "fresh"}),
     node("clue", "clue", "Workspace slugs are\nunique per owner", -640, 0, {}),
     node("prop", "proposal", "Split Members out\nof Workspaces", -640, 140, {"status": "PENDING"}),
     node("ws", "concept", "Workspaces", -280, 0, {"freshness": "fresh", "clues": 23, "footer": "touched 2d ago"}),
     node("f1", "file", "src/services/workspace.ts", 60, -120, {"inDocs": True, "importance": 0.92, "freshness": "fresh"}),
     node("f2", "file", "src/app/api/workspaces/\nroute.ts", 60, 120, {"inDocs": True, "importance": 0.81, "freshness": "fresh"}, height=88),
     node("fn", "function", "createWorkspace()", 400, -180, {"callers": 6}),
     node("slug", "function", "validateSlug()", 400, -60, {"callers": 2}),
     node("ep", "endpoint", "POST /api/workspaces", 400, 120, {}),
     node("dm", "datamodel", "Workspace", 700, -120, {})],
    [edge("pr", "ws", "TOUCHES", sides="right", color=G, strokeWidth=2.5),
     edge("clue", "ws", "RELEVANT_TO", sides="right", color="#fde68a", strokeWidth=2),
     edge("clue", "slug", "REFERENCES", sides="auto", fromSide="bottom", toSide="bottom", color="#fde68a", strokeWidth=1.2),
     edge("prop", "ws", "TARGETS", sides="right", color="#fdba74", strokeWidth=2),
     edge("ws", "f1", "MODIFIES 0.92", sides="right", color=G, strokeWidth=w(0.92)),
     edge("ws", "f2", "MODIFIES 0.81", sides="right", color=G, strokeWidth=w(0.81)),
     edge("f1", "fn", "CONTAINS", sides="right", color="#a78bfa", strokeWidth=2.5),
     edge("f1", "slug", "CONTAINS", sides="right", color="#a78bfa", strokeWidth=2.5),
     edge("f2", "ep", "CONTAINS", sides="right", color="#22d3ee", strokeWidth=2.5),
     edge("fn", "dm", "CALLS", sides="right", color="#a78bfa", strokeWidth=2),
     edge("ep", "fn", "CALLS", sides="auto", color="#22d3ee", strokeWidth=1.5)])

# ---------------------------------------------------------------- Tier 3: agent trace
add("08-trace-one-run", "Agent trace: one run", "Chat spawns a run; the run executes a workflow version; each step is a session; each tool call ACCESSED the nodes it read. 14 nodes.",
    [node("chat", "chat", "chat: fix workspace invites", -720, -40, {}),
     node("run", "run", "run r-2291", -420, -40, {"status": "OK", "footer": "score 0.86 · 3 steps"}),
     node("wf", "workflow", "workspace-triage", -420, -220, {"version": "v3"}),
     node("s1", "session", "step 1 · read", -120, -240, {"footer": "3 tool calls"}),
     node("s2", "session", "step 2 · patch", -120, -40, {"footer": "2 tool calls"}),
     node("s3", "session", "step 3 · reflect", -120, 160, {"footer": "1 tool call"}),
     node("t1", "toolcall", "search_concepts", 200, -320, {}),
     node("t2", "toolcall", "get_files_for_concept", 200, -220, {}),
     node("t3", "toolcall", "read_file", 200, -120, {}),
     node("t4", "toolcall", "edit_file", 200, -20, {}),
     node("t5", "toolcall", "save_clue", 200, 160, {}),
     node("c", "concept", "Workspaces", 540, -280, {"freshness": "fresh", "clues": 24, "footer": "cluesCount 23 → 24"}),
     node("f", "file", "src/services/workspace.ts", 540, -110, {"inDocs": True, "importance": 0.92, "freshness": "fresh"}),
     node("clue", "clue", "invites expire after 7d", 540, 160, {})],
    [edge("chat", "run", "SPAWNED", sides="right", color="#f472b6", strokeWidth=2.5),
     edge("run", "wf", "EXECUTED", sides="up", color="#a3e635", strokeWidth=2),
     edge("s1", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("s2", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("s3", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("t1", "s1", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t2", "s1", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t3", "s1", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t4", "s2", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t5", "s3", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t1", "c", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2.5),
     edge("t2", "c", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2),
     edge("t2", "f", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2),
     edge("t3", "f", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2.5),
     edge("t4", "f", "ACCESSED", sides="right", color="#fb923c", strokeWidth=3),
     edge("t5", "clue", "WROTE", sides="right", color="#fde68a", strokeWidth=2.5),
     edge("clue", "c", "RELEVANT_TO", sides="up", color="#fde68a", strokeWidth=2)])

add("09-trace-on-the-tree", "Agent trace: a run reading the tree", "The knowledge tree on the right; a run's steps and tool calls on the left, each pointing at the node it read. Reflection writes back. 16 nodes.",
    [node("repo", "repository", "stakwork/hive", 300, -440, {"footer": "5 concepts · 1 run today"}),
     node("ws", "concept", "Workspaces", 60, -260, {"freshness": "fresh", "clues": 24, "footer": "touched just now"}),
     node("runner", "concept", "Task Runner", 560, -260, {"freshness": "fresh", "clues": 17, "footer": "touched 5h ago"}),
     node("f1", "file", "src/services/workspace.ts", -80, -100, {"inDocs": True, "importance": 0.92, "freshness": "fresh"}),
     node("f2", "file", "src/app/api/workspaces/\nroute.ts", 210, -100, {"inDocs": True, "importance": 0.81, "freshness": "fresh"}, height=88),
     node("pool", "file", "src/lib/runner/pool.ts", 560, -100, {"inDocs": True, "importance": 0.87, "freshness": "fresh"}),
     node("fn", "function", "createWorkspace()", -80, 60, {"callers": 6}),
     node("ep", "endpoint", "POST /api/workspaces", 210, 60, {}),
     node("spawn", "function", "RunnerPool.spawn()", 560, 60, {"callers": 4}),
     node("run", "run", "run r-2291", -1260, -100, {"status": "OK", "footer": "workspace-triage v3 · 0.86"}),
     node("s1", "session", "step 1 · read", -940, -260, {"footer": "3 tool calls"}),
     node("s2", "session", "step 2 · patch", -940, -100, {"footer": "2 tool calls"}),
     node("s3", "session", "step 3 · reflect", -940, 60, {"footer": "1 tool call"}),
     node("t1", "toolcall", "search_concepts", -620, -260, {}),
     node("t3", "toolcall", "read_file", -620, -100, {}),
     node("t5", "toolcall", "save_clue", -620, 60, {})],
    [edge("repo", "ws", "0.95", color=G, strokeWidth=w(0.95)),
     edge("repo", "runner", "0.88", color=G, strokeWidth=w(0.88)),
     edge("ws", "f1", "MODIFIES 0.92", color=G, strokeWidth=w(0.92)),
     edge("ws", "f2", "MODIFIES 0.81", color=G, strokeWidth=w(0.81)),
     edge("runner", "pool", "MODIFIES 0.87", color=G, strokeWidth=w(0.87)),
     edge("f1", "fn", "CONTAINS", color="#a78bfa", strokeWidth=2.5),
     edge("f2", "ep", "CONTAINS", color="#22d3ee", strokeWidth=2.5),
     edge("pool", "spawn", "CONTAINS", color="#a78bfa", strokeWidth=2.5),
     edge("s1", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("s2", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("s3", "run", "IN_RUN", sides="left", color="#fbbf24", strokeWidth=2),
     edge("t1", "s1", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t3", "s2", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t5", "s3", "IN_SESSION", sides="left", color="#cbd5e1", strokeWidth=1.5),
     edge("t1", "ws", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2.5),
     edge("t3", "f1", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2.5),
     edge("t3", "fn", "ACCESSED", sides="right", color="#fb923c", strokeWidth=2),
     edge("t5", "ws", "WROTE clue", sides="right", color="#fde68a", strokeWidth=2.5)])

# ---------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
index_rows = []
for slug, title, subtitle, nodes, edges in DIAGRAMS:
    html = TEMPLATE.format(title=title, subtitle=subtitle, prelude=PRELUDE,
                           nodes="\n".join(nodes), edges="\n".join(edges), root_label=js(title))
    with open(os.path.join(OUT, f"{slug}.html"), "w") as fh:
        fh.write(html)
    index_rows.append((slug, title, subtitle, len(nodes), len(edges)))
    print(f"{slug}: {len(nodes)} nodes, {len(edges)} edges")

tiers = [("Concepts · 3-4 nodes", "0[1-4]"), ("Code · 7-10 nodes", "0[5-7]"), ("Agent trace · >12 nodes", "0[89]")]
cards = []
for slug, title, subtitle, n, e in index_rows:
    cards.append(f"""      <a class="card" href="{slug}.html">
        <img src="{slug}.png" alt="{title}" loading="lazy" />
        <div class="meta"><b>{title}</b><span>{subtitle}</span><small>{n} nodes · {e} edges</small></div>
      </a>""")
index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><title>Graph diagram series</title>
<style>
  body {{ margin: 0; background: #0a0a0a; color: #e2e8f0; font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; padding: 32px; }}
  h1 {{ font-size: 18px; margin: 0 0 6px; }} p {{ color: #94a3b8; margin: 0 0 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
  .card {{ display: block; border: 1px solid #1e293b; border-radius: 10px; overflow: hidden; background: #0f172a; text-decoration: none; color: inherit; }}
  .card:hover {{ border-color: #34d399; }}
  .card img {{ width: 100%; aspect-ratio: 16/10; object-fit: cover; display: block; background: #0a0a0a; }}
  .meta {{ padding: 12px 14px; display: grid; gap: 4px; }} .meta span {{ color: #94a3b8; }} .meta small {{ color: #64748b; }}
</style></head><body>
<h1>Everything goes in the graph</h1>
<p>Concepts (3-4 nodes) → Code (7-10) → Agent trace (&gt;12). Click a card for the interactive canvas; the big tree is <a href="knowledge-tree.html" style="color:#34d399">knowledge-tree.html</a>.</p>
<div class="grid">
{chr(10).join(cards)}
</div>
</body></html>
"""
with open(os.path.join(OUT, "index.html"), "w") as fh:
    fh.write(index)
print("index.html")
