---
description: "Redis beside the app, when it needs a cache or a queue to boot: one container, REDIS_URL at localhost."
parent: Pod Setup - Local Services
---
Redis beside the app, when it needs a cache or a queue to boot: one container, REDIS_URL at localhost.

- A `redis` compose service (`image: redis:7`), `ports: ["6379:6379"]`.
- The app's `REDIS_URL` is `redis://localhost:6379`; separate host and
  port variables get `localhost` and `6379`.
- Only when the app needs it to boot or for a core flow (sessions, a job
  queue the app waits on); a cache the code can do without stays off.
