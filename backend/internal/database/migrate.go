package database

import (
	"errors"
	"fmt"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/migrations"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"gorm.io/gorm"
)

// newMigrateInstance creates a new migrate instance using embedded migrations and the provided gorm.DB.
func newMigrateInstance(db *gorm.DB) (*migrate.Migrate, error) {
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("get sql.DB handle: %w", err)
	}

	driver, err := postgres.WithInstance(sqlDB, &postgres.Config{})
	if err != nil {
		return nil, fmt.Errorf("create postgres migration driver: %w", err)
	}

	sourceDriver, err := iofs.New(migrations.FS, ".")
	if err != nil {
		return nil, fmt.Errorf("create iofs source driver: %w", err)
	}

	m, err := migrate.NewWithInstance("iofs", sourceDriver, "postgres", driver)
	if err != nil {
		return nil, fmt.Errorf("create migrate instance: %w", err)
	}

	return m, nil
}

// Migrate applies all pending up migrations.
func Migrate(db *gorm.DB) error {
	m, err := newMigrateInstance(db)
	if err != nil {
		return fmt.Errorf("initialize migration runner: %w", err)
	}

	if err := m.Up(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("apply migrations: %w", err)
	}

	return nil
}

// MigrateDown rolls back all migrations.
func MigrateDown(db *gorm.DB) error {
	m, err := newMigrateInstance(db)
	if err != nil {
		return fmt.Errorf("initialize migration runner: %w", err)
	}

	if err := m.Down(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("rollback migrations: %w", err)
	}

	return nil
}

// MigrateStep rolls back or applies n migration steps.
func MigrateStep(db *gorm.DB, n int) error {
	m, err := newMigrateInstance(db)
	if err != nil {
		return fmt.Errorf("initialize migration runner: %w", err)
	}

	if err := m.Steps(n); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("migrate steps (%d): %w", n, err)
	}

	return nil
}

// GetMigrationVersion returns the current migration version and dirty flag.
func GetMigrationVersion(db *gorm.DB) (uint, bool, error) {
	m, err := newMigrateInstance(db)
	if err != nil {
		return 0, false, fmt.Errorf("initialize migration runner: %w", err)
	}

	version, dirty, err := m.Version()
	if err != nil {
		if errors.Is(err, migrate.ErrNilVersion) {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("get migration version: %w", err)
	}

	return version, dirty, nil
}

// ForceMigrationVersion sets the migration version and clears the dirty flag.
func ForceMigrationVersion(db *gorm.DB, version int) error {
	m, err := newMigrateInstance(db)
	if err != nil {
		return fmt.Errorf("initialize migration runner: %w", err)
	}

	if err := m.Force(version); err != nil {
		return fmt.Errorf("force migration version (%d): %w", version, err)
	}

	return nil
}

// Baseline initializes migration tracking on an existing database schema at the specified baseline version.
// If migration tracking is already initialized, it returns without error.
func Baseline(db *gorm.DB, baselineVersion int) error {
	m, err := newMigrateInstance(db)
	if err != nil {
		return fmt.Errorf("initialize migration runner: %w", err)
	}

	version, dirty, err := m.Version()
	if err != nil {
		if errors.Is(err, migrate.ErrNilVersion) {
			// No migration tracking table exists yet: force baseline version
			if err := m.Force(baselineVersion); err != nil {
				return fmt.Errorf("baseline migration version to %d: %w", baselineVersion, err)
			}
			return nil
		}
		return fmt.Errorf("check current migration version: %w", err)
	}

	if dirty {
		return fmt.Errorf("database is in dirty state at version %d; resolve before baselining", version)
	}

	return nil
}
