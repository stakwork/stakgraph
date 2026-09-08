# Stadeum: Graph-native agents that learn the whole system

Today we are launching **Stadeum**, an open-source platform for agents that know the whole system they work in. Frontier models are very good at the task in front of them; what they lack is the big picture. A senior engineer is more reliable than a model because the engineer knows the whole system: what connects to what, what has been tried, why things are the way they are. A model starts every task with none of that, and as agents take on whole codebases and whole bodies of law, the task itself becomes the easy part. Stadeum gives agents what the senior engineer has: a shared, persistent model of the whole system, stored as a graph. Every agent reads from it before acting and writes to it when done, so it grows past what any one person can hold. Our long-term aim is enterprise general intelligence, knowledge work at greater than human accuracy; this post is about the mechanism.

## Three failures

A capable model makes each of these when it can see only its task; the first is the running example for the rest of the post.

**The stale comment.** A comment describes how a function retries payments, and the description is two years out of date. The model trusts it and writes a caller that assumes exponential backoff against code that now retries on a fixed delay. Nothing in the file says the comment is wrong.

**The duplicate name.** Two repositories each contain a class called `PaymentProcessor`, written by different developers for different purposes. A language server sees one repository at a time, so the model that finds the wrong one gets no signal that another exists, and does accurate work in the wrong place.

**The missing relationship.** A federal regulation and a state regulation both bear on a required notice period, and each reads as complete on its own. The model that finds one document stops there. Neither document holds the answer; the relationship between them does.

In every case the model performed its task well and the system failed around it. Stadeum removes that class of failure with two rules.

## Two design rules

1. **The graph is the harness.** A harness is everything around the model at run time: its prompt, its tools, its skills, its context. In Stadeum every one of those is a node, and swarm assembles an agent by walking the graph and selecting the nodes for the task at hand. There is no configuration to drift out of sync with the knowledge, because the configuration is the knowledge. Acting shows that assembly for the payment-retry task.
2. **Read before acting, write when done.** Every agent checks its plan against the graph's structure and history before it acts, and writes what it learned back when it finishes. The same loop runs at every scale, and it is what makes the graph better with use rather than worse.

Four components implement those rules.

## Architecture

*[Figure 1: swarm at the center; hive, strut, and Jamie around it. Every inward arrow is a read, every outward arrow a write.]*

**Swarm** is the graph, federated so that each domain lives on its own server in its own vocabulary. **Strut** is the workflow engine, where a multi-step procedure is a versioned object that agents create, run, score, and improve. **Hive** is the work UI, where people turn feature plans into tasks, watch agents run them in cloud sandboxes, and steer. **Jamie** is the oracle agent that reads any graph, spawns any sub-agent, and gates every structural change.

For the engineer with five minutes: swarm, strut, and hive are open source under [license], and Jamie ships with the hosted version. The graph lives in Neo4j with vector search, and agents run against whichever model provider you configure, with the key on your side. Self-host with one `docker compose up` or use the hosted version at [hosted link]; the quickstart indexes one of your repositories into a local graph in [quickstart time].

The rest of the post follows the loop, from what is in the graph to how we measure whether any of it learns. The payment-retry comment comes back at the checkpoint, in the workflow, and in the reflection.

## The graph (swarm)

Everything Stadeum knows is a node. Every Concept, Code entity, Doc, PR, Eval, Prompt, Tool, Skill, and Workflow is one, and so is the Trace of every agent run, along with the Schemas that define each type. A Concept is a named idea in the domain, such as *payment retry policy*; a Trace is the record of one agent run. A graph that grows without bound needs discipline to stay legible, so the foundational layer of swarm is the ontology.

### Schema and placement

A meta-node called a Schema defines each node type: the properties expected on nodes of that type and the edges they may form. Swarm validates every node against its Schema, and an agent that attempts an invalid node receives a concise error describing how to correct it. Swarm gives agents one `create_triplet` tool for nodes and edges together, which guarantees that every new node is placed within the existing structure. At minimum, each Concept must have a parent Concept, yielding a hierarchy in which every node links to its category; the sole exception is the root Concept, which describes the domain as a whole. An agent that cannot say where a Concept belongs has not finished understanding it, and the tooling makes that visible at the moment of writing rather than at the moment of failure.

### Descriptions, embeddings, and code

Validation also opens the door to processing at creation time. Swarm generates a natural-language description of each new node with an LLM and embeds it, which yields a normalized embedding space across all nodes of a type: semantic search groups things by role rather than by vocabulary. For code, swarm ingests the repository's full history. Each merged pull request becomes a PR node, and swarm extracts the Concepts it introduces or touches from its intent and review. It parses the code with tree-sitter and resolves it through the language server, so every edge between Code entities is a resolved reference, and a name match alone never creates one. PageRank over those references gives every entity a structural importance score before any agent has looked at it.

### History

Because pull requests, evals, and Traces are nodes rather than metadata, swarm holds the history of a system and not only its current state, and computes staleness, contradiction, contested history, and importance from that record. Consider a node with forty neighbors. As a snapshot it is simply dense. With its history it is either a core abstraction that accumulated those edges over five years, a module that landed last month, or something contested, where edges keep being added and removed as people try to use it and back away. Those three call for very different handling, and the graph can tell them apart. The retry comment's Doc node has a history too: its last change predates three merged pull requests against the code it describes. Holding the history is one thing; reading it without drowning is the next.

## Reading (swarm)

A swarm graph holds far more than fits in a context window, so search alone cannot be the answer. Every node is discoverable by full-text or semantic search, but search only provides an entry point; full discovery requires walking, inspecting a node's neighbors and choosing which paths to explore. Swarm discloses neighbor information progressively: an agent first sees aggregate counts by neighbor and edge type, then names and descriptions, and only then reads chosen neighbors in full. Shape comes before content, so the agent knows how much is entangled with a node before spending any context on it. A sparsely connected Concept can be handled directly, while a dense one signals that the work should be split along its edges.

Even so, a single agent traversing a large graph is quickly overwhelmed, so graph agents spawn child walkers. A walker is a sub-agent that explores one pathway with its own context window and reports back which nodes are relevant. On a large task an agent routinely spawns dozens, each assigned a distinct slice of the knowledge work, and assembles a context window containing only what is relevant. Rather than being handed a context, the agent constructs one.

Walkers are also how the graph federates. Each domain, whether an organization's code or a body of law, lives on its own server with its own Schemas and its own consolidation loop, so a mess in one cannot leak into another. When a question spans two, the agent spawns a walker on the other server, which already knows how its graph is built and reports back the nodes that matter. The notice-period question is the federated case: a walker on the legal graph returns the `controls` edge from the state rule to the federal one. A review agent's reflection wrote that edge from a partner's memo with both regulation nodes as evidence, Jamie accepted it, and neither regulation has changed since. The duplicate `PaymentProcessor` is the single-domain case: search returns both nodes, and disclosure shows disjoint caller sets and different parent Concepts before the agent opens either.

## Reality checkpoints (swarm)

We call the read that precedes every action a reality checkpoint, and it is how the senior engineer's advantage becomes a tool call. Peter Naur observed in 1985 that a program is not its source text but the theory its builders hold of it: what it is for, why it is shaped the way it is, what would break if a piece changed. Once the builders leave, the program can only be modified by guesswork [1]. An agent arrives at every task in that condition, which is why reading the graph serves as verification. Before an agent acts on a belief, that a function retries on failure, that two names refer to one component, that a federal rule controls, it checks that belief against the graph. The check consults the node's current edges, the pull requests and Traces that wrote them, and its neighbors' timestamps, and comes back one of five ways:

1. **Confirmed and stable.** The edge exists and nothing near it has moved.
2. **Confirmed but stale.** The edge exists, but the region has not been touched while its neighbors moved.
3. **Contradicted.** The graph holds a different edge, and the checkpoint returns the one that disagrees.
4. **Contested.** The history shows oscillation, with the edge added and removed more than once.
5. **Unknown.** Nothing in the graph speaks to the belief, and the check returns the nearest nodes.

Each verdict implies a different next move, and PageRank tells the agent how much verification a belief deserves, since a wrong assumption about a leaf affects one thing while a wrong assumption about a hub affects everything downstream. This is the swarm client an agent step calls, on the running example:

```python
# Every call below is also recorded in the graph: the call becomes a node with
# an edge to each node it read, so the read itself is part of the record.

# Search is the entry point, not the answer
hits = await swarm.search("payment retry policy", kind="Concept")
node = hits[0]

# Shape before content: aggregate neighbor and edge counts
stats = await swarm.neighbor_stats(node.id)
# -> {Code: 14, Doc: 3, PR: 9, Trace: 22, Concept: 4}

# Fan out: walkers explore paths and report back relevant nodes
callers = await swarm.walk(node.id, "Find every caller that depends on the retry count.")
# -> [Code:billing.worker.charge, Code:checkout.retry_later, Code:refunds.replay]
docs = await swarm.walk(node.id, "Check whether docs for this Concept postdate its last change.")
# -> [Doc:payments/retry.md  last_changed=2024-06-02  code_changed=3x since, last 2026-08-19]

# Checkpoint: test the belief the plan depends on
verdict = await swarm.check(node.id, edge="retries_via", target=("Code", "ExponentialBackoff"))
# -> contradicted: retries_via -> FixedDelayRetry (PR #4812, 19 days ago)
```

Because the checkpoint came back *contradicted* with the pull request that changed the policy, the agent reads the implementation and writes the caller against the retry policy that exists. The stale comment cost one tool call instead of an incident.

## Acting (hive and strut)

This is where the first rule pays off. When a person in hive dispatches the payment-retry task, swarm assembles the agent that runs it. The task's Concept selects the Prompt fragments attached to it, the Tools are the Tool nodes whose Schemas match the domain, and the Skills are the Workflows that scored well on this shape of task before. Here that is a Prompt node for code triage, Tool nodes for the four calls above, a Skill node for the payments domain, and the context the walkers returned. Nothing about that agent is hand-maintained: the Prompt node's text is the workflow's `triagePrompt` default, and the promotion below is the write that updates it, so a better prompt written back last week is the prompt every agent gets this week. The thing an agent knows and the thing an agent *is* come from the same place, and the same process maintains both.

A hive task is one attempt at a problem, dispatched by a person or by Jamie. When the same shape of task recurs and there is an eval to score it, a dataset of cases with a rubric, an agent promotes it to a strut workflow, so the tenth attempt benefits from the nine before it. Each step of the workflow is a swarm agent that reads, checkpoints, acts, and reflects through the client above. The workflow and its versions are nodes in swarm, and so is every run. The run becomes a Trace with an edge to the exact version it ran and to its score, and every tool call in it becomes a node with an edge to each graph node it read, so the procedure and its track record sit together and the graph knows which knowledge each piece of work consumed.

In hive a feature plan decomposes into tasks, one per repository it touches, and each task's agent spawns its own walkers; Figure 2 shows the payment-retry plan across the billing, checkout, and refunds repositories.

*[Figure 2: hive, showing the payment-retry plan decomposed into tasks across the billing, checkout, and refunds repositories, with the swarm walkers spawned for each.]*

Here is the strut side as strut records it: the workflow, four lines from one run, the optimizer's proposal, and the promotion. Every parameter in the `params` block is a knob a run can override at any nesting depth without republishing.

```yaml
# 1. The workflow, version 2, as swarm stores it. The prompt is a param,
#    so the optimizer can change it without touching the steps.
name: payment-retry-triage
params:
  triagePrompt: "Read the implementation before its comment. Report the retry policy with the PR that last changed it."
steps:
  - id: locate
    type: agent
    tools: [swarm.search, swarm.neighbor_stats, swarm.walk]
    prompt: "{{params.triagePrompt}}"
  - id: verify
    type: agent
    tools: [swarm.check]
    input: "{{steps.locate.output}}"
  - id: patch
    type: agent
    tools: [repo.edit, repo.test]
    input: "{{steps.verify.output}}"
  - id: reflect
    type: agent
    tools: [swarm.create_triplet, swarm.add_evidence, swarm.create_task]
    input: "{{steps.patch.output}}"

# 2. Four of the forty lines one v2 run logged: the checkpoint verdict,
#    one step's cost, and what the scorer said the run missed.
{"runId":"1788825600000","path":"payment-retry-triage/locate","type":"step.start","stepType":"agent"}
{"runId":"1788825600000","path":"payment-retry-triage/verify","type":"tool.call","tool":"swarm.check","result":"contradicted"}
{"runId":"1788825600000","path":"payment-retry-triage/patch","type":"step.end","usage":"41k tokens","costUsd":0.19}
{"runId":"1788825600000","path":"payment-retry-triage","type":"run.end","score":0.52,"costUsd":0.42,"missing":["caller in refunds.replay","PR that set the fixed delay"]}

# 3. What the optimizer proposed after reading every miss across the dataset,
#    scored on the held-out set. Generation 3 won.
{"firstScore":0.52,"bestScore":0.86,"bestGen":3,"totalCost":4.12,
 "bestPrompt":"Read the implementation before its comment. Report the retry policy with the PR that last changed it, and name every caller that depends on the retry count.",
 "missing":[]}

# 4. The promotion. The param default lands in v3; v2 stays in the graph,
#    and rolling back is moving the pointer.
payment-retry-triage: v2 -> v3
```

The checkpoint block in the previous section is a v3 run, which is why it found `refunds.replay` and PR #4812; the v2 run logged here missed both. The `reflect` step is where the next section's reflection runs.

Look at the block again. The `missing` list is the capture: an observed failure turned into data the next step can read. A sweep over recent run logs for a signature like `command not found: yt-dlp` is a capture too, so a gap becomes an input rather than an anecdote in a chat transcript. `bestPrompt` is the proposal, written by an agent that read the misses across the whole dataset. 0.86 is the evaluation, one number on the held-out set with per-example detail behind it. `v2 -> v3` is the promotion. Strut runs those four beats at three layers, and only the thing promoted differs.

At the prompt layer it is a `params` value, promoted to the default in a new workflow version, and the loop runs on its own in minutes. At the environment layer it is the interpreters, CLIs, and libraries a step's shell can reach, held in a manifest the loop edits and promoted to a rebuilt image after a person reviews the two-line diff. At the structure layer it is the steps and their wiring, authored as a new workflow version from inside a run, so a workflow publishes its own successor. One eval scores all three, and one invariant holds: nothing self-modifies in place. A promotion is always a versioned, diffable artifact, which keeps every score attributable to a configuration; it is the strut analogue of "the configuration is the knowledge." The run is over, and what the agent learned has to go somewhere.

## Writing back (swarm)

Every agent ends its work with a reflection step, in which it adds to or updates any layer of the system: a Trace of what it did, a corrected Concept, an amended workflow step, a follow-up task. Reflection runs with a human in the loop or without one, and it is why nobody has to remember to maintain the graph. Here is the payment-retry agent's `reflect` step, each write validated against its Schema before it lands:

```python
trace = swarm.current_trace()

# What the walkers found and the graph did not yet hold: who depends on the retry count
for caller in callers:
    await swarm.create_triplet(
        subject=("Code", caller.name),
        predicate="depends_on_retry_count",
        object=("Concept", "payment retry policy"),
        evidence={"trace": trace.id},
    )

# The edge the checkpoint returned already exists; this run adds itself as evidence
await swarm.add_evidence(
    node.id, edge="retries_via", target=("Code", "FixedDelayRetry"),
    evidence={"pr": 4812, "trace": trace.id},
)

# The doc is wrong, not merely old: record the contradiction and queue the rewrite
await swarm.create_triplet(
    subject=("Doc", "payments/retry.md"),
    predicate="contradicts",
    object=("Code", "FixedDelayRetry"),
    evidence={"pr": 4812, "trace": trace.id},
)
await swarm.create_task("Rewrite payments/retry.md against FixedDelayRetry", about=docs[0].id)
```

The next agent that checks `retries_via` against `FixedDelayRetry` gets *confirmed and stable*, with the Trace that established it one edge away, and the one that asks who depends on the retry count gets three edges instead of a walk. What one walker discovers becomes durable structure for every future agent.

The same extends to the agents' own activity. Every graph-touching tool call is recorded with the nodes it read, and reflection rolls those reads up into one `READ_CONCEPT` edge from the Trace to each Concept. It carries a `rank` for how much the Concept mattered, an `evidence` field for what it supported, and a `contradicts` field when the agent found it wrong. Swarm reads three things off that record. A node that runs read and then cite gains weight in search; one they read and ignore loses it. Two nodes that runs always fetch together and never both use go to the dream cycle as a merge candidate. And a checkpoint that comes back *unknown* is a recorded gap, which the reflection that resolves it fills.

Some curation needs the whole graph at once, and a task agent is by design local, so the same loop runs a second way, as a dream cycle: a consolidation pass that runs nightly over the entire graph. Candidate generation is mechanical: *payment retry policy* and *webhook redelivery* have a description-embedding cosine of 0.88, share two pull requests, and overlap on a third of their files. An adjudicator agent reads both nodes with their evidence and returns `merge`, `supersedes`, `variant_of`, or `distinct`. Here it returns `variant_of`, and swarm writes a discriminator sentence into both nodes: "unlike webhook redelivery, this policy retries the charge itself and never re-sends the event." Discriminators are the point: contrastive sentences fix retrieval between near-matches where longer descriptions do not.

Each verdict has its write. `distinct` writes discriminators and records the pair; `variant_of` adds a `variant_of` edge as well; `supersedes` writes a `supersedes` edge from the newer node to the older and keeps the older as an alias. `merge` writes a `same_as` edge and points the loser's edges at the keeper: the loser's name becomes an alias, its PR list unions into the keeper, and it stays resolvable by its old id. Nothing is deleted under any verdict, and no later cycle re-asks an adjudicated pair.

What stops the graph from learning the wrong thing? Every write a reflection or a dream cycle makes passes the same Schema validation as any other. Every promotion is a version, so a bad one is undone by moving the pointer back. Proposals come from the aggregate and scores from a held-out set, so a fix for one example that hurts the rest loses. And graders, gold answers, and rubrics live in code outside anything an agent can edit, a producing agent never receives its own rubric, and an authoring agent never holds `bash`, so no agent grades its own homework. Those guards do not catch writes that contradict what people have decided; that is Jamie's job.

### Jamie

Jamie sorts writes into two kinds. Evidence, which is Traces, `READ_CONCEPT` edges, descriptions, staleness scores, and any edge that carries a pull request or a Trace as evidence, commits as soon as it passes validation. The reflection above committed on that rule, and so does a prompt-layer promotion, on its eval score. Structure, which is merges, new tools, new steps, Schema changes, environment and structure promotions, and any edge with no evidence behind it, is a proposal. Jamie holds each one with a diff, a staleness guard, and an audit trail until Jamie or a person decides it. How much review a proposal earns scales with the PageRank of what it touches, the same score that sets how carefully a belief is verified. Jamie clears a merge of two leaf Concepts alone; a merge touching a node with forty neighbors waits for a person.

One reconciliation shows the mechanism. A dream cycle flags the two Concepts the `PaymentProcessor` pull requests extracted, *payment processing* under the marketplace parent and under the billing parent: same name, description-embedding cosine of 0.91. The adjudicator sees what Reading showed, disjoint callers and different parents, and returns `variant_of`. The proposal reaches Jamie with a diff of one edge and two discriminators, a staleness guard that voids it if either node changes first, and an audit trail for the decision. Jamie checks it against what people have already said and finds a two-month-old review thread in which the marketplace team kept the two classes separate on purpose, because one processes charges and the other reconciles ledgers. The thread turns `variant_of` into a decided `distinct`: Jamie writes the discriminator into both nodes and links the thread as evidence, so no cycle proposes the pair again. When no record exists, Jamie asks, and the engineer's chat answer becomes a node: a decision, linked to the person, dated, and attached to the merge it authorized.

Jamie's reach covers every graph and every channel people talk on, and it has read what people have already said: conversation history, meeting transcripts, and the decisions recorded in review. This is the human-in-the-loop, and it is deliberately not a review queue for every agent output. People steer the system by doing what they already do: reviewing pull requests, judging evals, deciding in meetings, answering Jamie. Human judgment enters once, at the boundary, and the graph carries it to every task that follows. Whether that adds up to learning is a question of measurement.

## Does it learn?

Documentation drifts because keeping it current is a separate job that nobody's task depends on. The graph does not drift the same way. There is one structure and no second copy to fall behind it. Every write passes validation, including the writes reflection and dream cycles make. And because the history is kept, staleness, contradiction, and contest are facts swarm computes rather than things someone has to notice. Those reasons say why it can learn; the evals say whether it does.

Every eval's scorer returns an F2, weighting recall twice as heavily as precision, because the failure we are fixing is omission: the caller the agent did not find, the rule it did not check. It returns that number plus a `missing` list, the input to the next proposal, so the optimizer chases the misses that recur across the dataset rather than one task's quirks. The optimizer's fitness charges the score for cost, because a structural evolver otherwise discovers that adding agents raises scores. Every reported improvement comes from a held-out set the proposer never sees, and strut first runs the unchanged workflow several times to see how far the score moves on its own, so a challenger must beat the baseline by more than that. The payment-retry-triage dataset is 40 tasks, 30 train and 10 held out, each a repository, a stale comment, and the gold caller list; 0.52 to 0.86 above is its held-out curve.

Our strongest result comes from GAIA, a public benchmark of assistant tasks that need tools. On a five-task slice of its level 1, the baseline workflow scored 1/5, and one pass through the three layers took it to 5/5. Two tasks flipped on the prompt layer, both from rules about how to work rather than what to know: re-read the question and check the units; try a genuinely different approach before answering. The second rule made agents persist, one run then exhausted the step cap of 30 and took the whole batch down, and an agent raised the cap to 50 with an error fallback: a structural fix nobody asked for that everything after depended on. Two tasks then flipped on the environment layer because `pdftotext` and `yt-dlp` now existed in the image. The PDF task went from $0.40 to $0.08 a run; the video task went from $0.05 to $2.00, answered on step 50, the last the cap allows, and passed because the agent could finally attempt the work. The fifth passed at baseline and again after; [n reruns] of the unchanged workflow scored between 0/5 and 2/5, so one of the four flips is within that floor. Those five tasks are now a train set; the held-out number is [held-out score, n].

*[Figure 3: per-generation F2 for payment-retry-triage (30 train, 10 held out) and for the five-task GAIA slice (train) beside the held-out split of the 53-task sweep, with cost per run on the second axis.]*

## Next steps

Our work now is on the mechanisms that most directly decide whether the system learns. We are extending the churn history the checkpoint reads from Concept edges to every edge type. We are running dream cycles nightly across [N repositories] and one legal domain, and publishing the held-out curve from Figure 3 as it moves. In strut's structure layer, a workflow already publishes its own successor, and we are moving that from agent-interactive runs to scheduled headless runs with Jamie's review on every promotion. And we are running the full 53-task GAIA level-1 sweep on a held-out split, so the five-task result has company.

Naur's point was that the theory of a program lives in people and leaves with them. Stadeum's bet is that the theory can live in a structure that agents read before every task and repair after it, so it survives the people and improves with use. Accuracy at the scale of a task is a property of the model. Accuracy at the scale of a system is a property of the structure the model acts within, and that structure is what we are building.

The code is at [repository link]. Run the quickstart first: index one repository into a local swarm, ask for its stalest documented Concept, and see what the checkpoint says about the comment you trusted most.

---

[1] Peter Naur, "Programming as Theory Building," *Microprocessing and Microprogramming* 15 (1985).

<!--
Placeholders:
- [repository link] (Next steps)
- [license], [hosted link], [quickstart time] (Architecture); confirm Jamie's status ("ships with the hosted version")
- [N repositories] (Next steps)
- [n reruns] and [held-out score, n] (Does it learn?)
- Figure 1: swarm at the center; hive, strut, Jamie around it; inward arrows reads, outward arrows writes
- Figure 2: hive, the payment-retry plan across the billing, checkout, and refunds repositories, with walkers per task
- Figure 3: per-generation F2, payment-retry-triage train vs held-out and the GAIA slice vs the 53-task held-out split, cost on the second axis
-->
