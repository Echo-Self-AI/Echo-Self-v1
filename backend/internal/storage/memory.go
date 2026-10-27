package storage

import (
	"bytes"
	"context"
	"io"
	"sync"
)

type memoryObject struct {
	data        []byte
	contentType string
}

// MemoryStorage is an in-memory implementation of Storage for testing and local development.
type MemoryStorage struct {
	mu           sync.RWMutex
	objects      map[string]memoryObject
	FailUpload   error
	FailDownload error
	FailDelete   error
	FailExists   error
}

// NewMemoryStorage creates a new in-memory storage instance.
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		objects: make(map[string]memoryObject),
	}
}

func (m *MemoryStorage) Upload(ctx context.Context, key string, body io.Reader, size int64, contentType string) error {
	if m.FailUpload != nil {
		return m.FailUpload
	}
	if key == "" {
		return ErrInvalidKey
	}

	data, err := io.ReadAll(body)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.objects[key] = memoryObject{
		data:        data,
		contentType: contentType,
	}
	return nil
}

func (m *MemoryStorage) Download(ctx context.Context, key string) (io.ReadCloser, int64, string, error) {
	if m.FailDownload != nil {
		return nil, 0, "", m.FailDownload
	}
	if key == "" {
		return nil, 0, "", ErrInvalidKey
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	obj, exists := m.objects[key]
	if !exists {
		return nil, 0, "", ErrObjectNotFound
	}

	return io.NopCloser(bytes.NewReader(obj.data)), int64(len(obj.data)), obj.contentType, nil
}

func (m *MemoryStorage) Delete(ctx context.Context, key string) error {
	if m.FailDelete != nil {
		return m.FailDelete
	}
	if key == "" {
		return ErrInvalidKey
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.objects, key)
	return nil
}

func (m *MemoryStorage) Exists(ctx context.Context, key string) (bool, error) {
	if m.FailExists != nil {
		return false, m.FailExists
	}
	if key == "" {
		return false, ErrInvalidKey
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	_, exists := m.objects[key]
	return exists, nil
}

func (m *MemoryStorage) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.objects)
}

func (m *MemoryStorage) GetObjects() map[string][]byte {
	m.mu.RLock()
	defer m.mu.RUnlock()
	res := make(map[string][]byte, len(m.objects))
	for k, v := range m.objects {
		res[k] = v.data
	}
	return res
}

var _ Storage = (*MemoryStorage)(nil)
