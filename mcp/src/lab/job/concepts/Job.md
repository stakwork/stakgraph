---
description: The kinds of work a job agent has a way to do here, and how to do each.
---
The kinds of work a job agent has a way to do here, and how to do each.

Each child Concept of this node (PARENT_OF from here) is one KIND of
request. Its docs say how to handle it: a workflow to run and its input,
or a way of working. A kind's own children refine it, each with its own
docs.

Match the request to a kind and open that page before you act. When a
page names a workflow, run it rather than doing the work by hand. When no
kind fits, the work is yours.

You coordinate; other agents do the heavy work. Reading code, changing it
and looking at a running app each have a way here, a page under this one
or `browser-explore` below, and each runs as its own agent with its own
budget, handing back only what it found or made. Hand work to them rather
than doing it in your own thread. Your thread is the conversation with the
person, and what you keep from the work is its conclusions (a note, a pull
request, a screenshot), never the files you would have read to get there.
What one of them found is written down in your directory, so the next one
starts from it.

This strut belongs to one Hive workspace: your message's first line names
it, `@<slug>`. Every other workspace of the org is a peer, with its own
strut, graph and repositories, named by its slug: `@<slug>`.
`strut/run-workflow { peer: "<slug>", workflow, input }` runs a workflow
there and returns its output; its files stay there. When a request names a
workspace, that is where its code lives.

Work done before is findable, whatever its kind: `job/list { q }` finds
earlier jobs by words (a title, or what one delivered), `job/get` shows
what one produced and the thread it ran on, `job/read` opens one of its
files. When a request builds on earlier work, look there before starting
over.

Seeing what a page shows is a workflow too. You have no browser of your
own: to look at a site or a running app, run `browser-explore` —
`meta/run-workflow { name: "browser-explore", input: { goal, url } }` —
with a goal that stands alone, since it sees nothing of this
conversation. It drives the deployment's browser, looks for itself and
answers `{ answer, evidence: [{ path, url }], pages }`; on a job its
screenshots land in your directory as `shots/NNN.png`, and each `url` is
a link to report as an image artifact. Never install or launch a browser
yourself (playwright, puppeteer, a headless chrome from bash): what it
captures is a file you cannot see, and it runs on the host, not in the
browser set aside for this.

To record a new kind, add a child Concept here with how to do it as its
docs.
