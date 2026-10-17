---
description: "MySQL or MariaDB beside the app: a container with fixed dev credentials, reached at 127.0.0.1."
parent: Pod Setup - Local Services
---
MySQL or MariaDB beside the app: a container with fixed dev credentials, reached at 127.0.0.1.

- A `mysql` compose service (`image: mysql:8`, or `mariadb` when the
  repository uses it), `ports: ["3306:3306"]`, a named volume for
  `/var/lib/mysql`, and fixed `MYSQL_DATABASE`, `MYSQL_USER`,
  `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD`.
- The app's database URL or host settings point at `127.0.0.1:3306` with
  those values — never `localhost`, which a MySQL client takes for its
  Unix socket.
- Migrations in pre-start, as for any database.
