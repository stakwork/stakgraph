---
name: cocounsel-legal
description: >
  CoCounsel Legal delivers comprehensive Westlaw Deep Research reports with inline, linked
  citations to Westlaw and Practical Law sources.
tags: [legal]
source: https://github.com/anthropics/claude-for-legal
source_ref: 4a6c651889c97cc9140580363c73e0eb17379c2b
license: Apache-2.0
attribution: Anthropic — claude-for-legal
---

# cocounsel-legal

## Running outside Claude Code

This pack was written for the Claude Code plugin runtime. You are not running in
it. The guidance below is otherwise accurate — adapt these points as you read:

- **Slash commands are unavailable.** Where a skill says to run
  `/cocounsel-legal:some-command`, there is no such command. Either perform that step's
  work directly, or load the correspondingly-named skill with
  `load_skill("cocounsel-legal/some-command")` and follow it.
- **Matter workspaces are off.** Treat `## Matter workspaces` as disabled
  (`Enabled: ✗`), which is the documented default. Work at the practice level and
  skip matter-switching, per-matter folders, and cross-matter rules entirely.
- **Plugin config paths do not exist.** Ignore instructions to read or write
  `~/.claude/plugins/config/...`. Write outputs to the working directory unless
  the user names a destination, and never assume a prior run's files are present.
- **This file is the practice-level CLAUDE.md.** When a skill refers to "the
  practice-level CLAUDE.md" or "this plugin's CLAUDE.md", it means the sections
  below. There is no separate file to open.
- **Customization is unsaved.** The profile sections below hold upstream defaults,
  not this user's real practice. Where a decision depends on firm-specific
  configuration that is clearly still a placeholder, ask rather than assume.
- **Plugin hooks, sub-agents, and MCP connectors are absent.** Anything relying on
  them needs to be done with the tools you actually have.

---



