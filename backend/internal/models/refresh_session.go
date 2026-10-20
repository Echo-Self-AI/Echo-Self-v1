package models

import (
	"time"

	"github.com/google/uuid"
)

type RefreshSession struct {
	ID        uuid.UUID  `gorm:"type:uuid;primaryKey" json:"-"`
	UserID    uuid.UUID  `gorm:"type:uuid;not null;index" json:"-"`
	TokenHash string     `gorm:"type:char(64);uniqueIndex;not null" json:"-"`
	ExpiresAt time.Time  `gorm:"not null;index" json:"-"`
	RevokedAt *time.Time `json:"-"`
	CreatedAt time.Time  `json:"-"`
}

func (RefreshSession) TableName() string {
	return "refresh_sessions"
}
