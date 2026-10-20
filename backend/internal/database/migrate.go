package database

import (
	"fmt"

	"github.com/Echo-Self-AI/Echo-Self-v1/backend/internal/models"
	"gorm.io/gorm"
)

func Migrate(db *gorm.DB) error {
	if err := db.AutoMigrate(
		&models.User{},
		&models.RefreshSession{},
	); err != nil {
		return fmt.Errorf("migrate database schema: %w", err)
	}

	return nil
}
