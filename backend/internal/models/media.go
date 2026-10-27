package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

const (
	MediaStatusReady           = "ready"
	MediaStatusDeletionPending = "deletion_pending"
	MediaStatusDeleted         = "deleted"

	MediaCategoryVoiceSample = "voice_sample"
	MediaCategoryFaceSample  = "face_sample"
	MediaCategoryGeneral     = "general"
)

type Media struct {
	ID               uuid.UUID      `gorm:"type:uuid;primaryKey" json:"id"`
	UserID           uuid.UUID      `gorm:"type:uuid;not null;index" json:"user_id"`
	Category         string         `gorm:"type:varchar(50);not null" json:"category"`
	StorageKey       string         `gorm:"type:varchar(255);uniqueIndex;not null" json:"storage_key"`
	ContentType      string         `gorm:"type:varchar(100);not null" json:"content_type"`
	FileSize         int64          `gorm:"not null" json:"file_size"`
	OriginalFilename string         `gorm:"type:varchar(255);not null" json:"original_filename"`
	Status           string         `gorm:"type:varchar(50);not null;default:'ready'" json:"status"`
	CreatedAt        time.Time      `json:"created_at"`
	UpdatedAt        time.Time      `json:"updated_at"`
	DeletedAt        gorm.DeletedAt `gorm:"index" json:"-"`
}

func (Media) TableName() string {
	return "media"
}

func (m *Media) BeforeCreate(tx *gorm.DB) error {
	if m.ID == uuid.Nil {
		m.ID = uuid.New()
	}
	return nil
}
