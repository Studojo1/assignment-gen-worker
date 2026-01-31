package blob

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blockblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/container"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/sas"
)

// Client uploads files to blob storage.
type Client interface {
	Upload(ctx context.Context, jobID, localPath string) (downloadURL string, err error)
}

// AzureClient uses real Azure Blob Storage.
type AzureClient struct {
	containerClient *container.Client
	accountName     string
	accountKey      string
	containerName   string
}

// NewAzureClient creates an Azure Blob Storage client.
func NewAzureClient(accountName, accountKey, containerName string) (Client, error) {
	if accountName == "" {
		return nil, fmt.Errorf("AZURE_STORAGE_ACCOUNT_NAME required")
	}
	if accountKey == "" {
		return nil, fmt.Errorf("AZURE_STORAGE_ACCOUNT_KEY required")
	}

	cred, err := azblob.NewSharedKeyCredential(accountName, accountKey)
	if err != nil {
		return nil, fmt.Errorf("create credential: %w", err)
	}

	containerURL := fmt.Sprintf("https://%s.blob.core.windows.net/%s", accountName, containerName)
	containerClient, err := container.NewClientWithSharedKeyCredential(containerURL, cred, nil)
	if err != nil {
		return nil, fmt.Errorf("create container client: %w", err)
	}

	ctx := context.Background()
	_, err = containerClient.Create(ctx, nil)
	if err != nil {
		// Container might already exist, ignore
	}

	return &AzureClient{
		containerClient: containerClient,
		accountName:     accountName,
		accountKey:      accountKey,
		containerName:   containerName,
	}, nil
}

func (c *AzureClient) Upload(ctx context.Context, jobID, localPath string) (string, error) {
	blobName := fmt.Sprintf("%s/%s", jobID, filepath.Base(localPath))
	blobClient := c.containerClient.NewBlockBlobClient(blobName)

	file, err := os.Open(localPath)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	_, err = blobClient.UploadStream(ctx, file, &blockblob.UploadStreamOptions{})
	if err != nil {
		return "", fmt.Errorf("upload: %w", err)
	}

	// Generate SAS token for secure access (1 hour expiration)
	cred, err := azblob.NewSharedKeyCredential(c.accountName, c.accountKey)
	if err != nil {
		return "", fmt.Errorf("create credential for SAS: %w", err)
	}

	permissions := sas.BlobPermissions{Read: true}
	sasQueryParams, err := sas.BlobSignatureValues{
		Protocol:      sas.ProtocolHTTPS,
		StartTime:     time.Now().UTC(),
		ExpiryTime:    time.Now().UTC().Add(1 * time.Hour),
		Permissions:   permissions.String(),
		ContainerName: c.containerName,
		BlobName:      blobName,
	}.SignWithSharedKey(cred)
	if err != nil {
		return "", fmt.Errorf("generate SAS token: %w", err)
	}

	// Return blob URL with SAS token
	blobURL := blobClient.URL()
	return fmt.Sprintf("%s?%s", blobURL, sasQueryParams.Encode()), nil
}

// LocalstackClient uses LocalStack for local Azure Blob Storage emulation.
type LocalstackClient struct {
	containerClient *container.Client
	endpoint        string
	accountName     string
	containerName   string
}

// NewLocalstackClient creates a LocalStack blob storage client.
func NewLocalstackClient(endpoint, accountName, accountKey, containerName string) (Client, error) {
	if accountName == "" {
		accountName = "test"
	}
	if accountKey == "" {
		accountKey = "test"
	}

	cred, err := azblob.NewSharedKeyCredential(accountName, accountKey)
	if err != nil {
		return nil, fmt.Errorf("create credential: %w", err)
	}

	// LocalStack S3 emulation - use container name as bucket name
	// For LocalStack S3, the URL format is: http://localstack:4566/{bucket}
	// We use the containerName as the S3 bucket name
	endpoint = strings.TrimSuffix(endpoint, "/")
	// LocalStack S3: bucket name is the container name, account is ignored
	containerURL := fmt.Sprintf("%s/%s", endpoint, containerName)
	containerClient, err := container.NewClientWithSharedKeyCredential(containerURL, cred, nil)
	if err != nil {
		return nil, fmt.Errorf("create container client: %w", err)
	}

	// Create container (S3 bucket) if it doesn't exist
	ctx := context.Background()
	_, err = containerClient.Create(ctx, &container.CreateOptions{
		Metadata: map[string]*string{},
	})
	if err != nil {
		// Check if error is because container already exists (ignore those)
		errStr := err.Error()
		if !strings.Contains(errStr, "ContainerAlreadyExists") && 
		   !strings.Contains(errStr, "409") &&
		   !strings.Contains(errStr, "already exists") &&
		   !strings.Contains(errStr, "BucketAlreadyOwnedByYou") {
			// Log but don't fail - container might already exist
			fmt.Printf("Warning: container create returned error (may already exist): %v\n", err)
		}
	}

	return &LocalstackClient{
		containerClient: containerClient,
		endpoint:        endpoint,
		accountName:     accountName,
		containerName:   containerName,
	}, nil
}

func (c *LocalstackClient) Upload(ctx context.Context, jobID, localPath string) (string, error) {
	// Ensure container exists before uploading
	_, err := c.containerClient.Create(ctx, &container.CreateOptions{
		Metadata: map[string]*string{},
	})
	if err != nil {
		// Ignore if container already exists
		errStr := err.Error()
		if !strings.Contains(errStr, "ContainerAlreadyExists") && 
		   !strings.Contains(errStr, "409") &&
		   !strings.Contains(errStr, "already exists") {
			// Log warning but continue
			fmt.Printf("Warning: container create check failed: %v\n", err)
		}
	}

	blobName := fmt.Sprintf("%s/%s", jobID, filepath.Base(localPath))
	blobClient := c.containerClient.NewBlockBlobClient(blobName)

	file, err := os.Open(localPath)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	_, err = blobClient.UploadStream(ctx, file, &blockblob.UploadStreamOptions{})
	if err != nil {
		// LocalStack S3 returns 200 OK with no body, which Azure SDK treats as error
		// But the upload actually succeeded (LocalStack logs show s3.PutObject => 200)
		// For LocalStack, we can ignore this specific error and proceed
		errStr := err.Error()
		if strings.Contains(errStr, "Response contained no body") || 
		   strings.Contains(errStr, "ERROR CODE UNAVAILABLE") {
			// Upload succeeded on LocalStack side, just SDK parsing issue
			// Proceed to generate download URL
			err = nil
		} else {
			return "", fmt.Errorf("upload: %w", err)
		}
	}

	// LocalStack S3 public URL format: http://localhost:4566/{bucket}/{key}
	// For external browser access, use localhost instead of service name
	externalEndpoint := c.endpoint
	if strings.Contains(externalEndpoint, "localstack:") {
		externalEndpoint = strings.Replace(externalEndpoint, "localstack:", "localhost:", 1)
	}
	// For S3, bucket is containerName, no account in path
	downloadURL := fmt.Sprintf("%s/%s/%s", externalEndpoint, c.containerName, blobName)
	return downloadURL, nil
}
