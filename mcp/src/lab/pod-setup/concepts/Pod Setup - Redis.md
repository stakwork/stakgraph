---
description: "Redis beside the app, when it needs a cache or a queue to boot: one container, REDIS_URL at 127.0.0.1."
parent: Pod Setup - Local Services
---
Redis beside the app, when it needs a cache or a queue to boot: one container, REDIS_URL at 127.0.0.1.

- A `redis` compose service (`image: redis:7`), `ports: ["6379:6379"]`.
- The app's `REDIS_URL` is `redis://127.0.0.1:6379`; separate host and
  port variables get `127.0.0.1` and `6379`.
- Only when the app needs it to boot or for a core flow (sessions, a job
  queue the app waits on); a cache the code can do without stays off.
