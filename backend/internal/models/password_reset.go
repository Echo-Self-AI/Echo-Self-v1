package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type PasswordReset struct {
	ID        uuid.UUID  `gorm:"type:uuid;primaryKey" json:"-"`
	UserID    uuid.UUID  `gorm:"type:uuid;not null;index" json:"-"`
	TokenHash string     `gorm:"type:char(64);uniqueIndex;not null" json:"-"`
	ExpiresAt time.Time  `gorm:"not null;index" json:"-"`
	UsedAt    *time.Time `json:"-"`
	CreatedAt time.Time  `json:"-"`
}

func (PasswordReset) TableName() string {
	return "password_resets"
}

func (p *PasswordReset) BeforeCreate(tx *gorm.DB) error {
	if p.ID == uuid.Nil {
		p.ID = uuid.New()
	}
	return nil
}
