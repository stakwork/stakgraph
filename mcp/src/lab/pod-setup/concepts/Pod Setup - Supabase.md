---
description: "A Supabase app in a pod: the Supabase CLI's local stack, never hand-wired containers, with the repository's migrations applied."
parent: Pod Setup - Local Services
---
A Supabase app in a pod: the Supabase CLI's local stack, never hand-wired containers, with the repository's migrations applied.

- No compose services for Supabase: the CLI starts its own Postgres, auth,
  storage and REST containers. Never wire those up by hand.
- The pre-start command runs the CLI from the repository root (where
  `supabase/` lives): `npx -y supabase start && npx -y supabase db reset
  --local` when `supabase/migrations/*.sql` exist. With no
  `supabase/config.toml`, `npx -y supabase init --force` first. When the
  app's directory is a subdirectory, the command begins with
  `cd /workspaces/<repo> &&`.
- Schema SQL kept elsewhere in the repository is copied into
  `supabase/migrations/` with a timestamped name before the reset; no
  migration is invented when none exists. Never `supabase db query -f`
  for a multi-statement file.
- The app's env: `SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL` is
  `http://localhost:54321`; the public (anon / publishable) key is
  `sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH`, the server-only secret
  (service role) key `sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz`, under the
  names the repository reads.
