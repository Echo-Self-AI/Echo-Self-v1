# Database Migrations Guide

EchoSelf uses versioned database migrations powered by `golang-migrate/migrate/v4`.
SQL migration files are stored in `migrations/` and embedded directly into the Go application binaries using `embed.FS`.

---

## 1. Directory Structure

```
backend/
├── cmd/
│   └── migrate/
│       └── main.go           # CLI entrypoint for database migrations
├── internal/
│   └── database/
│       ├── migrate.go        # Programmatic migration runner
│       └── migrate_test.go   # Integration tests for migrations
├── migrations/
│   ├── 000001_initial_schema.up.sql
│   ├── 000001_initial_schema.down.sql
│   └── migrations.go         # embed.FS definition
└── docs/
    └── migrations.md         # This documentation
```

---

## 2. CLI Commands

The migration CLI is accessible via:

```bash
go run ./cmd/migrate <command> [args]
```

### Commands:

| Command | Description | Example |
|---|---|---|
| `up` | Apply all pending migrations | `go run ./cmd/migrate up` |
| `down` | Roll back all migrations | `go run ./cmd/migrate down` |
| `step <n>` | Apply `n` steps (+n for up, -n for rollback) | `go run ./cmd/migrate step 1` or `step -1` |
| `version` | Display current migration version and dirty status | `go run ./cmd/migrate version` |
| `force <v>` | Set version explicitly and clear dirty state after fixing a failure | `go run ./cmd/migrate force 1` |
| `baseline [v]` | Mark an existing database as up-to-date at version `v` (default: 1) without re-running SQL statements | `go run ./cmd/migrate baseline 1` |

---

## 3. Creating Future Migrations

1. Choose the next sequential 6-digit version number (e.g. `000002`).
2. Create two SQL files in the `migrations/` directory:
   - `migrations/000002_<description>.up.sql`
   - `migrations/000002_<description>.down.sql`
3. Write standard, idempotent PostgreSQL DDL:
   - Always use `IF NOT EXISTS` or explicit constraint names.
   - Ensure rollbacks in `.down.sql` safely reverse `.up.sql`.
4. Run migrations locally:
   ```bash
   go run ./cmd/migrate up
   ```
5. Verify schema changes in your test database before committing.

---

## 4. Safe Baselining for Existing Databases

When introducing versioned migrations to an existing database (such as an existing production or staging database that was created with `AutoMigrate`):

1. **Do NOT run destructive rollback or table recreation.**
2. Run the baseline command:
   ```bash
   DATABASE_URL="<db_connection_string>" go run ./cmd/migrate baseline 1
   ```
3. This creates the `schema_migrations` table and sets version to `1` without re-executing `000001_initial_schema.up.sql`.
4. Subsequent migrations (version 2+) will apply normally on top of the baselined schema.

---

## 5. Handling Failed Migrations

If a migration fails mid-execution:
1. `golang-migrate` marks the database as **dirty** (`dirty = true`).
2. Further migration operations will be blocked to prevent schema corruption.
3. **Resolution Procedure**:
   - Inspect the database error and identify the failed statement.
   - Manually fix or roll back the failed SQL statement in the database.
   - Run `force` to reset the version to the last known good version (e.g., `go run ./cmd/migrate force <good_version>`).
   - Fix the migration script in code, then re-run `go run ./cmd/migrate up`.

---

## 6. Production Deployment Procedure

1. Before deploying a new application binary with schema changes:
   - Run `go run ./cmd/migrate up` as a pre-deploy step or CI/CD deployment task.
   - Alternatively, the application runs `database.Migrate(db)` on startup if configured for automatic unapplied migration execution.
2. Verify version:
   ```bash
   go run ./cmd/migrate version
   ```
3. Deploy application containers.
