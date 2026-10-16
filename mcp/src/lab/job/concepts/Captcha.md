---
description: When a browser step in the job's pod is blocked by a captcha. Ask the person in Jamie to tap tiles or type text, on an allowlisted site only.
parent: Job
---
When a browser step in the job's pod is blocked by a captcha. Ask the person in Jamie to tap tiles or type text, on an allowlisted site only.

## When

A browser step in the job's pod (agent-browser) is blocked by a captcha. You
decide that, from what the page shows. A quick hint: check the page's
`iframe[src]` for `google.com/recaptcha`, `recaptcha.net`, `hcaptcha.com` or
`challenges.cloudflare.com`. It is a hint only, never a gate: a captcha
without those iframes is still a captcha, and those iframes without a
challenge are not.

## Allowlist

Before you ask, the page's site must be on the workspace's captcha
allowlist. The person's Jamie must accept it: the answer route refuses
disallowed sites. If it refuses, stop and report `captcha_site_not_allowed`.
Never try to solve a captcha any other way, and never use a solving service.

## Browser recipe

Drive agent-browser in the job's pod through `pod/agent` with a standalone
task. It runs headless; there is no Xvfb. Use one session for the whole
captcha: name `captcha-<job>`, opened with `--idle-timeout 12m`. Call it S.

1. Every round starts with `agent-browser --session S get url`. If it says
   `about:blank`, reopen the page first.
2. Find the challenge iframe's box: `eval` over the iframes (for reCAPTCHA
   the one whose title contains `challenge`), or `get box`. Boxes are in CSS
   px. Record it as `box` = `{x, y, width, height}`.
3. Take a viewport screenshot to an ABSOLUTE path (relative paths land
   somewhere you will not find). The viewport shot is 1:1 with CSS px.
4. Crop it to the box with ImageMagick:
   `convert shot.png -crop WxH+X+Y +repage /tmp/staklink-files/captcha/round-N.png`.
   Python PIL and `bc` are not installed.
5. Bring the crop into the job: `pod/file { control, sealed, path:
   "captcha/round-N.png", to: "captcha/round-N.png" }`.

## Manifest

Write `captcha/round-N.json`:

    { "image": "captcha/round-N.png", "instruction": "<what the challenge asks>",
      "site": "<the page's site>", "grid": { "rows": 3, "cols": 3 },
      "mode": "tap" | "text" | "either", "round": N, "maxRounds": 5,
      "askedAt": "<ISO time>", "expiresAt": "<askedAt + 10 minutes>",
      "box": { "x": 0, "y": 0, "width": 0, "height": 0 } }

`grid` is optional (only for tile grids). List it in `artifacts` as
`{ id: "captcha", kind: "captcha", title, path: "captcha/round-N.json" }`;
the id is the same every round. Then END THE TURN with `ask`: "Captcha needs
you in Jamie". The page stays open in the pod between turns.

## Answer turn

The next turn's prompt is a line `[captcha-answer] {json}` with `{ round,
taps?: [{ nx, ny }], text?, skip? }`; `nx`, `ny` are fractions 0-1 of the
cropped image.

- For each tap, click page x = `box.x + nx * box.width`, y = `box.y + ny *
  box.height`, rounded to integers:
  `agent-browser --session S mouse move X Y`, then `mouse down left`, then
  `mouse up left`.
- For text, click the input, then `keyboard type` the text EXACTLY as given:
  case preserved, never trimmed.
- Do not move the page between the screenshot and the clicks. Re-measure the
  box before clicking and shift every click by any change.

Then re-screenshot and look:

- Solved: continue the task. Finish with `agent-browser close` on the
  session, and `pod/release` only if the job no longer needs the pod.
- A new round appeared: do round + 1, up to 5 rounds.
- `skip`, 5 rounds used, or no answer within 10 minutes (`expiresAt`
  passed): stop, report "captcha not solved", and close the session.
- The challenge expired (reCAPTCHA says it expires in two minutes): re-open
  it and send a fresh round.

## Tested facts

On Google's demo page, https://www.google.com/recaptcha/api2/demo, headless
works, and clicking by x/y into the cross-origin iframe works. Tiles have no
refs (only buttons do), so coordinates are required.
