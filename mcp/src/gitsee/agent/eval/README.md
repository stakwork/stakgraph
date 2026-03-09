# Services Agent Eval + GEPA Optimization

Evolve the services agent prompts using GEPA (Gradient-free Evolution of Prompts and Agents).

## How it works

```
                  ┌─────────────┐
                  │  Opus       │  "these prompts scored low on X,
                  │  reflects   │   here's an improved version..."
                  └──────┬──────┘
                         │ new prompts
                         v
  ┌──────────┐    ┌─────────────┐    ┌──────────────┐
  │ training │───>│  Sonnet     │───>│ deterministic │──> score
  │ repos    │    │  runs agent │    │ scoring       │
  └──────────┘    └─────────────┘    └──────────────┘
                         │
                    repeat until
                    budget exhausted
```

**Two LLMs are used:**
- **Opus** -- reflection + merge (analyzes failures, generates better prompts)
- **Sonnet** -- runs the agent itself (cheaper, one call per repo per generation)

**Scoring is fully deterministic** -- no LLM involved. It parses the generated configs and compares structurally against your gold standards (service names, env vars, ports, paths).

## File layout

```
eval/
  prompts/
    explorer.md          <-- system prompt (GEPA evolves this)
    final_answer.md      <-- tool description (GEPA evolves this)
  train/
    stakwork--hive/      <-- {owner}--{repo}
      pm2.config.js      <-- the correct output
      docker-compose.yml <-- the correct output
      notes.txt          <-- optional
    someorg--app/
      pm2.config.js
      docker-compose.yml
  val/                   <-- optional separate validation set
  runs/                  <-- optimization results saved here
```

## Step 1: Add training data

```bash
mkdir eval/train/stakwork--hive
# paste the correct pm2.config.js and docker-compose.yml in there
```

## Step 2: Score current prompts

```bash
npx tsx eval/run.ts --eval
```

## Step 3: Optimize

```bash
npx tsx eval/run.ts --optimize
```

Options:
```
--max-evals 30              budget (default 20)
--merge                     enable crossover of top candidates
--reflection-model opus     model for reflection (default: opus)
--quiet                     less output
```

Each run saves to `eval/runs/{timestamp}/`:
```
eval/runs/2026-03-09-14-30-22/
  explorer.md        best evolved system prompt
  final_answer.md    best evolved tool description
  summary.txt        score + metadata
  history.json       per-generation scores
```

## Step 4: Compare runs

```bash
npx tsx eval/run.ts --list
```

## Step 5: Apply the best one

```bash
npx tsx eval/run.ts --apply eval/runs/2026-03-09-14-30-22
```

This writes the evolved prompts back into `prompts/services.ts`.

## Scoring

Deterministic, no LLM. Parses generated configs and compares against gold:

| Dimension | Weight | What |
|-----------|--------|------|
| format | 10% | Both files produced? |
| pm2_structure | 20% | Service count + names match |
| docker_structure | 20% | Docker services match |
| env_vars | 15% | Required env vars present + critical values |
| app_service | 15% | App service has exact required structure |
| cwd | 10% | Working directories correct |
| frontend_naming | 10% | Has a service named "frontend" |

## Env vars

```
ANTHROPIC_API_KEY=sk-...
```

## Future: per-flag prompts

Currently the prompts are one big blob covering all project types (web, Android, Ruby, etc). Plan is to split into flag-based variants:

```
eval/
  prompts/
    base/                    <-- shared core (always included)
      explorer.md
      final_answer.md
    react/                   <-- appended when flag="react"
      explorer.md
      final_answer.md
    android/                 <-- appended when flag="android"
      explorer.md
      final_answer.md
  train/
    react/                   <-- training repos per flag
      stakwork--hive/
    android/
      someorg--myapp/
```

Each flag gets its own GEPA evolution track: `--optimize --flag react`. The base prompt can be frozen or co-evolved. This keeps each variant simpler and lets the scoring focus on repos of that type only.
