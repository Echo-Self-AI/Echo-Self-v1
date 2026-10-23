package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

const (
	ConsentStatusGranted = "granted"
	ConsentStatusRevoked = "revoked"
	ConsentStatusExpired = "expired"
)

type ConsentRecord struct {
	ID            uuid.UUID  `gorm:"type:uuid;primaryKey" json:"id"`
	UserID        uuid.UUID  `gorm:"type:uuid;not null;index" json:"user_id"`
	ConsentType   string     `gorm:"type:varchar(50);not null" json:"consent_type"`
	Status        string     `gorm:"type:varchar(20);not null" json:"status"`
	PolicyVersion string     `gorm:"type:varchar(20);not null" json:"policy_version"`
	GrantedAt     time.Time  `gorm:"not null" json:"granted_at"`
	RevokedAt     *time.Time `json:"revoked_at,omitempty"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
}

func (ConsentRecord) TableName() string {
	return "consent_records"
}

func (c *ConsentRecord) BeforeCreate(tx *gorm.DB) error {
	if c.ID == uuid.Nil {
		c.ID = uuid.New()
	}
	return nil
}

type ConsentAuditLog struct {
	ID            uuid.UUID `gorm:"type:uuid;primaryKey" json:"id"`
	ConsentID     uuid.UUID `gorm:"type:uuid;not null;index" json:"consent_id"`
	UserID        uuid.UUID `gorm:"type:uuid;not null;index" json:"user_id"`
	Action        string    `gorm:"type:varchar(50);not null" json:"action"`
	ConsentType   string    `gorm:"type:varchar(50);not null" json:"consent_type"`
	PolicyVersion string    `gorm:"type:varchar(20);not null" json:"policy_version"`
	IPAddress     string    `gorm:"type:varchar(45)" json:"ip_address,omitempty"`
	UserAgent     string    `gorm:"type:varchar(255)" json:"user_agent,omitempty"`
	CreatedAt     time.Time `gorm:"not null" json:"created_at"`
}

func (ConsentAuditLog) TableName() string {
	return "consent_audit_logs"
}

func (a *ConsentAuditLog) BeforeCreate(tx *gorm.DB) error {
	if a.ID == uuid.Nil {
		a.ID = uuid.New()
	}
	return nil
}
