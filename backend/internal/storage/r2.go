package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// R2Config holds the required configuration for Cloudflare R2 storage.
type R2Config struct {
	AccountID       string
	AccessKeyID     string
	SecretAccessKey string
	BucketName      string
	Endpoint        string
	Region          string
}

// R2Storage implements Storage for Cloudflare R2 using AWS SDK for Go v2.
type R2Storage struct {
	client     *s3.Client
	bucketName string
}

// NewR2Storage creates a new R2Storage instance.
func NewR2Storage(ctx context.Context, cfg R2Config) (*R2Storage, error) {
	if cfg.BucketName == "" {
		return nil, errors.New("r2: bucket name is required")
	}
	if cfg.AccessKeyID == "" || cfg.SecretAccessKey == "" {
		return nil, errors.New("r2: access key id and secret access key are required")
	}

	endpoint := cfg.Endpoint
	if endpoint == "" {
		if cfg.AccountID == "" {
			return nil, errors.New("r2: either endpoint or account id must be provided")
		}
		endpoint = fmt.Sprintf("https://%s.r2.cloudflarestorage.com", strings.TrimSpace(cfg.AccountID))
	}

	region := cfg.Region
	if region == "" {
		region = "auto"
	}

	credProvider := credentials.NewStaticCredentialsProvider(
		cfg.AccessKeyID,
		cfg.SecretAccessKey,
		"",
	)

	awsCfg, err := awsconfig.LoadDefaultConfig(ctx,
		awsconfig.WithRegion(region),
		awsconfig.WithCredentialsProvider(credProvider),
	)
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		o.BaseEndpoint = aws.String(endpoint)
		o.UsePathStyle = true
	})

	return &R2Storage{
		client:     client,
		bucketName: cfg.BucketName,
	}, nil
}

// Upload streams an object to the private R2 bucket.
func (r *R2Storage) Upload(ctx context.Context, key string, body io.Reader, size int64, contentType string) error {
	if strings.TrimSpace(key) == "" {
		return ErrInvalidKey
	}

	input := &s3.PutObjectInput{
		Bucket:        aws.String(r.bucketName),
		Key:           aws.String(key),
		Body:          body,
		ContentType:   aws.String(contentType),
		ContentLength: aws.Int64(size),
	}

	_, err := r.client.PutObject(ctx, input)
	if err != nil {
		return fmt.Errorf("r2 upload %q: %w", key, err)
	}

	return nil
}

// Download streams an object from the private R2 bucket.
func (r *R2Storage) Download(ctx context.Context, key string) (io.ReadCloser, int64, string, error) {
	if strings.TrimSpace(key) == "" {
		return nil, 0, "", ErrInvalidKey
	}

	input := &s3.GetObjectInput{
		Bucket: aws.String(r.bucketName),
		Key:    aws.String(key),
	}

	output, err := r.client.GetObject(ctx, input)
	if err != nil {
		var notFound *s3types.NoSuchKey
		if errors.As(err, &notFound) {
			return nil, 0, "", ErrObjectNotFound
		}
		return nil, 0, "", fmt.Errorf("r2 download %q: %w", key, err)
	}

	var size int64
	if output.ContentLength != nil {
		size = *output.ContentLength
	}

	var contentType string
	if output.ContentType != nil {
		contentType = *output.ContentType
	}

	return output.Body, size, contentType, nil
}

// Delete removes an object from the private R2 bucket.
func (r *R2Storage) Delete(ctx context.Context, key string) error {
	if strings.TrimSpace(key) == "" {
		return ErrInvalidKey
	}

	input := &s3.DeleteObjectInput{
		Bucket: aws.String(r.bucketName),
		Key:    aws.String(key),
	}

	_, err := r.client.DeleteObject(ctx, input)
	if err != nil {
		return fmt.Errorf("r2 delete %q: %w", key, err)
	}

	return nil
}

// Exists checks if an object exists in the private R2 bucket.
func (r *R2Storage) Exists(ctx context.Context, key string) (bool, error) {
	if strings.TrimSpace(key) == "" {
		return false, ErrInvalidKey
	}

	input := &s3.HeadObjectInput{
		Bucket: aws.String(r.bucketName),
		Key:    aws.String(key),
	}

	_, err := r.client.HeadObject(ctx, input)
	if err != nil {
		var notFound *s3types.NotFound
		if errors.As(err, &notFound) {
			return false, nil
		}
		var noSuchKey *s3types.NoSuchKey
		if errors.As(err, &noSuchKey) {
			return false, nil
		}
		return false, fmt.Errorf("r2 exists %q: %w", key, err)
	}

	return true, nil
}

var _ Storage = (*R2Storage)(nil)
