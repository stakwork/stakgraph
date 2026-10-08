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
