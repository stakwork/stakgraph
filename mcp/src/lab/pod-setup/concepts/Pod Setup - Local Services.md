---
description: "What runs beside the app in a pod, and what is mocked instead — the compose file's services, and the placeholders for everything else."
parent: Pod Setup
---
What runs beside the app in a pod, and what is mocked instead — the compose file's services, and the placeholders for everything else.

**A container only for what the app needs** to boot or for its core local
flows: the database it persists to, a cache or a queue it cannot start
without. An env var alone shows an integration exists, not that a local
service is needed.

**Everything else is mocked or switched off**: the repository's own mock or
offline mode when it has one (a `USE_MOCKS`, `MOCK_*`, `OFFLINE` or
`*_ENABLED` flag — turn it on), a disabled flag, or a deterministic
placeholder value. Cloud services — payments, email, auth providers,
analytics, LLM APIs, a hosted database's dashboard — never get a container
and never a real key: `mock-stripe-key`, `mock-resend-key`, and no
`your-project.supabase.co`-style URL that blocks the boot. The env var is
still set, so the code does not crash on a missing key.

**A service beside the app**:

- publishes its port (`ports: ["5432:5432"]`), and the app reaches it at
  `localhost` and that port, never at the service's name;
- gets fixed dev credentials, the same values in the app's env;
- keeps its data in a named volume when it has state.

**A datastore with its own local CLI** runs through the CLI, not as
compose services: its page says how.

The pages under this one (PARENT_OF) are the services, one each.
