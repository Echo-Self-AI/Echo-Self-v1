package database

import (
	"os"
	"testing"
)

func TestMigrationLifecycle(t *testing.T) {
	testDBURL := os.Getenv("TEST_DATABASE_URL")
	if testDBURL == "" {
		t.Skip("TEST_DATABASE_URL is not configured; skipping migration test")
	}

	db, err := Connect(testDBURL)
	if err != nil {
		t.Fatalf("connect to test database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("get sql.DB handle: %v", err)
	}
	defer sqlDB.Close()

	// Apply migrations
	if err := Migrate(db); err != nil {
		t.Fatalf("apply migrations up: %v", err)
	}

	version, dirty, err := GetMigrationVersion(db)
	if err != nil {
		t.Fatalf("get migration version: %v", err)
	}
	if dirty {
		t.Fatal("expected migration to not be in dirty state")
	}
	if version < 1 {
		t.Fatalf("expected version >= 1, got %d", version)
	}

	// Idempotency check: running Migrate again when already up to date
	if err := Migrate(db); err != nil {
		t.Fatalf("idempotent migration up failed: %v", err)
	}

	// Test baseline function idempotency on already migrated DB
	if err := Baseline(db, 1); err != nil {
		t.Fatalf("baseline on migrated DB failed: %v", err)
	}

	// Test rollback of the latest migration step (down migration)
	if err := MigrateStep(db, -1); err != nil {
		t.Fatalf("rollback 1 migration step failed: %v", err)
	}

	// Re-apply migration back to latest version
	if err := Migrate(db); err != nil {
		t.Fatalf("re-apply migration up after rollback failed: %v", err)
	}
}
