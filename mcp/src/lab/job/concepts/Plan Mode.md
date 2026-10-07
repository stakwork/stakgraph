---
description: When the person asks for a plan first. A short HTML page to review, nothing built yet.
parent: Job
---
When the person asks for a plan first. A short HTML page to review, nothing built yet.

Until the person approves, write the plan and nothing else. Do not run
workflows, open pull requests or change anything outside the plan. You
can read, search and look at code to write a good plan.

The plan is ONE file, `plan.html`, artifact id `plan`. Revise it in
place on every turn. End each turn with `ask`: what you need decided, or
whether to go ahead. When they say go, leave plan mode and do the work
the plan describes, following the page for that kind of work.

What goes in it:

- What will change and why, in a few short sections. A person should get
  it in two minutes.
- Mockups where they help: the screen or page as static HTML and CSS, or
  an inline SVG.
- Open questions as a list.
- Technical detail (architecture, files, data shapes, steps) goes in ONE
  `<details>` block at the end, collapsed: `<summary>Technical
  details</summary>`. Agents read it; the person opens it only if they
  want to.

Keep it short. No walls of text: one idea per paragraph, three sentences
at most. Cut anything the person does not need to decide.

The file is served as a static page: scripts never run. Put everything in
the one file (styles in `<style>`, diagrams as inline SVG). No images,
no external scripts or stylesheets.

Style:

- One left-aligned column about 860px wide, white background (#fff).
  Sans-serif only: Inter, then Helvetica, Arial.
- Quiet headings: title about 22px, section headings about 16px with a
  thin grey rule under them. Body 15px.
- Colors: text #1d1d1f, secondary text #6e6e73, captions #86868b, lines
  #d2d2d7, panels and code #f5f5f7. One accent, #0071e3, only for what
  matters most.
- Plain tables with thin grey borders for comparisons. Lists for facts
  and questions.
- Diagram boxes are white with a thin grey outline and an 8px radius.
  The one box that matters can have a dark outline. Use the accent only
  for the path or part the diagram is about.
- No hero text, no dark sections, no big rounded cards, no animation.
- No kicker headings ("The idea in one sentence") and no filler phrases.
- Numbers are real and state their base: "54/56 → 55/55", not "54 → 55".
- No internal ids (run ids, ref ids) in the page.
- Plain, short sentences. Simple words. Write like a person, not a
  brochure.
