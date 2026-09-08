# Stadeum: Graph-native agents that learn the whole system

Today, we're launching **Stadeum**, an open-source enterprise intelligence platform. Frontier models are now very good at the task they're given; what they're missing is the big picture. The reason a senior engineer is more reliable than a model isn't intelligence. It's that the engineer knows the whole system: what connects to what, what has been tried, why things are the way they are. A model starts every task with none of that, and as agents take on larger projects — whole codebases, whole organizations, whole bodies of law — the task itself becomes the easy part. Stadeum's objective is Enterprise General Intelligence: a system that completes any knowledge work task at greater than human accuracy. We believe the way there is to give agents what the senior engineer has — a shared, persistent model of the whole system. Every agent reads from it before acting and adds to it when done, so it grows past what any one person could hold. We've centered our design on graph data structures as the common language for that shared memory.

That shared memory is also the harness. Our agents are dynamically constructed using the principles of recursive language models, but every element of the harness is fetched at run time from the graph instead of from files. Code, prompt elements, workflow steps, tool descriptions, context, and even logs are stored as nodes, which means the thing an agent knows and the thing an agent *is* come from the same place and are maintained by the same process.

Stadeum is built around two abstractions:

1. **The graph is the harness.** An agent is assembled at run time by walking the graph: its prompt, its tools, its skills, and its context are all nodes, selected for the task at hand. There is no configuration to drift out of sync with the knowledge, because the configuration is the knowledge.
2. **Read before acting, write when done.** Every agent checks its plan against the graph's structure and history before it acts, and writes what it learned back when it finishes. The same loop runs at every scale, from a single task to system-wide consolidation, and it is what makes the graph better with use rather than worse.

Stadeum is fully open source: [repository link].

## Architecture

Four components make up the platform, and all of them are organized around the graph.

**swarm** is the graph itself, federated across servers so that each domain — an organization's code, a body of law, a general knowledge base — is held on its own server in its own vocabulary. It also carries the machinery that lives closest to the graph: the agent harnesses, the self-improvement loops, and the agents that watch log streams.

**strut** is the workflow engine. Complex, multi-step procedures are objects that agents create, curate, run, and optimize with loops, rather than plans reinvented on each attempt. A workflow is a node in swarm and each run of it is a trace, so the procedure and its track record are inspectable in the same place.

**hive** is where the work happens. Feature plans become tasks, tasks become code in cloud sandboxes, and the whole picture is visible across many repositories and initiatives at once. It is also where humans watch, approve, and steer, and where swarms and workflows are controlled.

**Jamie** is the oracle: the one agent whose context is the entire system. Jamie can read any graph, spawn any sub-agent, and communicate on any channel, and every proposal to change the system passes through Jamie on its way to becoming structure. This is where the platform meets human judgment.

*[Figure 1: swarm at the center; hive, strut, and Jamie around it. Every inward arrow is a read, every outward arrow a write.]*

The rest of this post follows the loop. We start with what is in the graph, then how an agent reads it, does its work, and writes back, and finally why the result compounds.

## Swarm: The Graph

Everything Stadeum knows is a node. Concepts, code entities, documents, pull requests, evals, prompts, tools, skills, workflows, and the traces of every agent run live in the same graph, along with the Schemas that define each of them. A graph that will grow to hold an unbounded number of concepts needs discipline to stay legible, to agents and to humans, so the foundational layer of swarm is the ontology.

### Schema and placement

Each node type is formally defined by a meta-node called a Schema, which specifies the properties expected on nodes of that type and the edges they are permitted to form. Every node created in the graph is validated against its Schema, and when an agent attempts to create an invalid node it receives a concise error describing how to correct it. Nodes can also require edges at creation time. Rather than exposing separate "create node" and "create edge" tools, agents use a single `create_triplet` tool, which guarantees that every new node is placed within the existing structure. At minimum, each Concept must have a parent Concept, yielding a hierarchy in which every node is linked to its category or neighborhood; the sole exception is the root Concept, which describes the domain of the graph as a whole. An agent that cannot say where a concept belongs has not finished understanding it, and the tooling makes that visible at the moment of writing rather than at the moment of failure.

### Descriptions, embeddings, and code

Validation also opens the door to processing at creation time. Each new node receives a natural-language description generated by an LLM, and that description is embedded, which yields a normalized embedding space across all nodes of a given type: semantic search groups things by role rather than by vocabulary. For code, ingestion works from the repository's history rather than from a snapshot. Each merged pull request becomes a node, the Concepts it introduces or touches are extracted from its intent and review, and the code is parsed with tree-sitter and resolved through the language server, so that edges between code entities are resolved references rather than name matches. PageRank over those references gives every entity a structural importance score before any agent has looked at it.

### History

Because pull requests, evals, and traces are nodes rather than metadata, swarm holds the history of a system and not only its current state, and staleness, contradiction, and importance are all computed from that record. Consider a node with forty neighbors. Viewed as a snapshot it is simply dense; viewed with its history it is either a core abstraction that accumulated those edges over five years, a module that landed last month, or something contested, where edges keep being added and removed as people try to use it and back away. Those three call for very different handling, and the graph can tell them apart.

### Federation

The graph is federated, and each can have its own set of schemas, or share schemas with other domains. When a question spans two of them, an agent spawns a graph walker on the other server; that walker already knows how its own graph is built and reports back the nodes that matter. Each domain keeps its own vocabulary and its own consolidation loop, and a mess in one cannot leak into another.

## Reading the Graph: Reality Checkpoints

Retrieval from a graph this size has its own challenges. Many swarm graphs hold far more than fits in a single context window — a software graph spanning every repository in an organization, or a legal graph covering an entire regulatory domain — so search alone cannot be the answer. Every node is discoverable by full-text or semantic search, but search only provides an entry point; full discovery requires walking, inspecting a node's neighbors and choosing which paths to explore.

To keep that tractable, neighbor information is progressively disclosed. An agent first sees aggregate statistics over a node's neighbor and edge types, can then filter to view names and descriptions, and only then selects specific neighbors to read in full. Shape comes before content, so the agent knows how much is entangled with a node before it has spent any context on it: a sparsely connected concept can be handled directly, while a dense one signals that the work should be split along its edges. Even so, a single agent traversing a large graph is quickly overwhelmed, which is why graph agents spawn child walkers, sub-agents that explore specific pathways and report back which nodes are relevant. On large or complex tasks an agent routinely spawns dozens, each assigned a distinct slice of the knowledge work, and assembles a context window containing only what is relevant. Rather than being handed a context, the agent constructs one.

Peter Naur observed in 1985 that a program is not its source text but the theory its builders hold of it — what it is for, why it is shaped the way it is, what would break if a piece changed — and that once they leave, the program can only be modified by guesswork [1]. An agent arrives at every task in that condition, which is why reading the graph is not retrieval but verification. Before an agent acts on a belief — that a function retries on failure, that two names refer to one component, that a federal rule controls — it checks that belief against the graph's structure and history. The check comes back one of five ways: confirmed and stable; confirmed but stale, meaning the region hasn't been touched while its neighbors moved; contradicted, with the edge that disagrees; contested, meaning the history shows oscillation; or unknown, with the nearest nodes. Each verdict implies a different next move, and importance tells the agent how much verification a belief deserves in the first place, since a wrong assumption about a leaf affects one thing while a wrong assumption about a high-PageRank node affects everything downstream. We call these reality checkpoints, and they are how the senior engineer's advantage becomes a tool call.

```python
# Search is the entry point, not the answer
hits = await swarm.search("payment retry policy", kind="Concept")
node = hits[0]

# Shape before content: aggregate neighbor and edge statistics
stats = await swarm.neighbors(node.id)
# -> {Code: 14, Doc: 3, PR: 9, Trace: 22, Concept: 4}

# Fan out: walkers explore paths and report back relevant nodes
callers = swarm.walk(node.id, "Find every caller that depends on the retry count.")
docs    = swarm.walk(node.id, "Check whether docs for this Concept postdate its last change.")

# Checkpoint: test the belief the plan depends on
verdict = await swarm.check(node.id, edge="retries_via", target="ExponentialBackoff")
# -> contradicted: retries_via -> FixedDelayRetry (PR #4812, 19 days ago)
```

## Doing the Work: Hive and Strut

Work itself happens in our web UI called Hive. A feature plan is broken into tasks, each task is coded in a cloud sandbox, and the results are visible across every repository and initiative the organization is running, so a change in one place is seen in the context of everything else in flight. Hive is also where people are: they watch tasks progress, approve or redirect them, and control the swarms and workflows underneath.

Where a task is one of many attempts at the same kind of problem, it runs through strut. A workflow in strut is a first-class object, created, curated, run, and optimized by agents, and because both the workflow and each of its runs are nodes in swarm, an agent choosing how to approach a task can see not only the procedure but how it performed the last several times it was used. Loops in strut refine the procedure across those runs, so the tenth attempt at a class of problem benefits from the nine before it.

*[Figure 2: hive, showing a feature plan decomposed into tasks across three repositories, with the swarm walkers spawned for each.]*

## Writing Back: Reflection and Dream Cycles

Every agent can end its work with a reflection step, in which it adds to or updates any layer of the system: a trace of what it did, a new or corrected Concept, an amended workflow step, a follow-up task. Reflection can run with a human in the loop or without one, and it is why the graph is not a static artifact that someone must remember to maintain. Knowledge discovered by one walker becomes durable structure available to every future agent, and the system's understanding of a domain compounds rather than being rediscovered with each task.

This extends to the agents' own activity. Traces are nodes with edges linking each task to the Concepts it retrieved, so over time the graph accumulates a record of which knowledge was used for which work. That record is computed over: it reveals high-value nodes, stale regions, and gaps, and turns usage itself into a signal for curation.

Some of that curation cannot be done by an agent focused on a task, because it requires seeing the whole graph at once. Deciding that two distant Concepts are duplicates, noticing that a region has gone quiet while everything around it moved, or detecting that two parts of the graph now contradict each other all need global scope, and a task agent is by design local. So the same read-act-reflect loop runs a second way, as a dream cycle: a consolidation pass over the entire graph that detects contradictions, rescores staleness and importance, merges what should be merged, fills gaps, and proposes new Concepts, prompts, tools, and workflows. Every write a dream cycle makes passes through the same Schema validation as every other write, which is what allows agents to maintain the graph without corrupting it.

## Jamie: Where the System Meets Human Judgment

Proposals from reflection and dream cycles do not become structure on their own. They filter through Jamie, the highest-level agent in the system. Jamie sees every graph and every top-level document, talks with people directly, and has read what they have already said — the conversation history, the meeting transcripts, the decisions recorded in review — so a proposal is reconciled against human intent before it is accepted. When a proposal needs evidence, Jamie spins up investigation agents on log streams or knowledge bases and gets it.

This is the human-in-the-loop, and it is deliberately not a review queue for agent outputs. People steer the system by doing what they already do — reviewing pull requests, judging evals, making decisions in meetings, talking to Jamie — and Jamie turns that into structure that every future agent reads before it acts. Human judgment enters once, at the boundary, and the graph carries it to every task that follows.

## Why It Learns

Documentation drifts because keeping it current is a separate job that nobody's task depends on. Swarm does not drift the same way, for four reasons. Everything is in the graph, so there is one structure for knowledge, process, work, and the record of all three, and no second copy to fall out of sync. Every write is validated, including the writes made by reflection and dream cycles, so agents maintaining the graph are bound by the same discipline as agents building it. History is kept, so staleness and contradiction are computed facts rather than things someone has to notice. And human judgment enters once, through Jamie, and propagates to every task instead of being re-applied to every output. Each pass through the loop leaves the graph more complete and more trusted, and what is relevant but not yet known shrinks with use.

## What It Produces

The failures we set out to prevent are the ones a capable model makes when it can only see its task. Three of them, and what an agent sees in swarm instead:

A comment describes how a function retries payments, and the description is two years out of date. The doc node for that Concept has a last-changed date that predates three changes to the implementation it describes, so it is already marked stale when the agent arrives; the checkpoint on the belief the comment produced returns *stale*, and the agent reads the implementation rather than its description.

Two repositories each contain a class called `PaymentProcessor`, written by different developers for different purposes. In swarm they are two nodes with disjoint caller sets and different parent Concepts. Search returns both, and progressive disclosure shows the difference before the agent has opened either.

A federal regulation and a state regulation both bear on a required notice period, and each reads as complete on its own. The Concept node for the notice requirement has edges to both regulations and an edge between them recording that the state rule is more stringent and controls; that edge is what the walk returns. Neither document contains the answer. The relationship does.

Evals are nodes too, and their scores across dream cycles are how we measure whether the graph is getting better.

*[Figure 3: eval accuracy over successive dream cycles for a code domain and a legal domain.]*

## Next Steps

Reality checkpoints and dream cycles are where our work is concentrated. The graph, the workflow engine, the coding UI, and Jamie are in daily use; the two mechanisms that most directly determine whether the system learns are the ones with the most room to improve, and we expect the largest gains in accuracy to come from there. Accuracy at the scale of a task is a property of the model. Accuracy at the scale of a system is a property of the structure the model acts within, and that structure is what we are building.

---

[1] Peter Naur, "Programming as Theory Building," *Microprocessing and Microprogramming* 15 (1985).

---

Placeholders to fill: the repository link, three figures, and real eval numbers under Figure 3 — the post is noticeably stronger with even one chart. The API names in the code block (`swarm.search`, `swarm.neighbors`, `swarm.walk`, `swarm.check`) are illustrative and should be swapped for your actual tool signatures; the shape of the example (search → stats → walkers → checkpoint verdict) is the part that matters.