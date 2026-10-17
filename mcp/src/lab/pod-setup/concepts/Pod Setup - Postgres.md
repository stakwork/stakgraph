---
description: "PostgreSQL beside the app: a postgres container with fixed dev credentials, and the app's DATABASE_URL at 127.0.0.1."
parent: Pod Setup - Local Services
---
PostgreSQL beside the app: a postgres container with fixed dev credentials, and the app's DATABASE_URL at 127.0.0.1.

- A `postgres` compose service (`image: postgres:16`, or the major
  version the repository names), `ports: ["5432:5432"]`, a named volume for
  `/var/lib/postgresql/data`, and fixed `POSTGRES_DB`, `POSTGRES_USER`,
  `POSTGRES_PASSWORD`.
- The app's `DATABASE_URL` (or whatever the code reads) is
  `postgresql://<user>:<password>@127.0.0.1:5432/<db>` with those values.
- Migrations run before the app: a pre-start command. With Prisma,
  `sleep 15 && npx prisma generate && npx prisma migrate deploy` when
  `prisma/migrations` exists, and `sleep 15 && npx prisma generate && npx
  prisma db push` when it does not: deploy only applies migrations, so
  with none it makes no tables. Never `prisma migrate dev`, which prompts.
  Other ORMs: their own migrate command (`alembic upgrade head`,
  `bin/rails db:prepare`, `npx drizzle-kit migrate`, or `npx drizzle-kit
  push` with no migrations folder).
- A seed script the repository has goes in post-start.
