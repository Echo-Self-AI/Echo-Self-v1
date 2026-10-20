package database

import (
	"os"
	"testing"

	"gorm.io/gorm"
)

func SetupTestDB(t *testing.T) *gorm.DB {
	t.Helper()

	databaseURL := os.Getenv("TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("TEST_DATABASE_URL is not configured; skipping integration test")
	}

	db, err := Connect(databaseURL)
	if err != nil {
		t.Fatalf("connect to test database: %v", err)
	}

	if err := Migrate(db); err != nil {
		sqlDB, _ := db.DB()
		if sqlDB != nil {
			_ = sqlDB.Close()
		}
		t.Fatalf("migrate test database: %v", err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("get test database handle: %v", err)
	}

	t.Cleanup(func() {
		if err := sqlDB.Close(); err != nil {
			t.Errorf("close test database: %v", err)
		}
	})

	return db
}
