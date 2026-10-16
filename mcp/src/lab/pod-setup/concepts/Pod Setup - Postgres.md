---
description: "PostgreSQL beside the app: a postgres container with fixed dev credentials, and the app's DATABASE_URL at localhost."
parent: Pod Setup - Local Services
---
PostgreSQL beside the app: a postgres container with fixed dev credentials, and the app's DATABASE_URL at localhost.

- A `postgres` compose service (`image: postgres:16`, or the major
  version the repository names), `ports: ["5432:5432"]`, a named volume for
  `/var/lib/postgresql/data`, and fixed `POSTGRES_DB`, `POSTGRES_USER`,
  `POSTGRES_PASSWORD`.
- The app's `DATABASE_URL` (or whatever the code reads) is
  `postgresql://<user>:<password>@localhost:5432/<db>` with those values.
- Migrations run before the app: a pre-start command. With Prisma,
  `sleep 15 && npx prisma generate && npx prisma migrate deploy` — never
  `prisma migrate dev`, which prompts. Other ORMs: their own migrate
  command (`alembic upgrade head`, `bin/rails db:prepare`, `npx drizzle-kit
  migrate`).
- A seed script the repository has goes in post-start.
