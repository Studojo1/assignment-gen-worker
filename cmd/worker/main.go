package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"

	"github.com/studojo/assignment-gen-worker/internal/blob"
	"github.com/studojo/assignment-gen-worker/internal/generator"
)

func main() {
	rabbitURL := os.Getenv("RABBITMQ_URL")
	if rabbitURL == "" {
		rabbitURL = "amqp://guest:guest@localhost:5672/"
	}
	resultsExchange := os.Getenv("RESULTS_EXCHANGE")
	if resultsExchange == "" {
		resultsExchange = "cp.results"
	}
	azureAccountName := os.Getenv("AZURE_STORAGE_ACCOUNT_NAME")
	azureAccountKey := os.Getenv("AZURE_STORAGE_ACCOUNT_KEY")
	azureContainerName := os.Getenv("AZURE_STORAGE_CONTAINER_NAME")
	if azureContainerName == "" {
		azureContainerName = "assignments"
	}
	useLocalstack := os.Getenv("USE_LOCALSTACK") == "true"
	localstackEndpoint := os.Getenv("LOCALSTACK_ENDPOINT")
	if localstackEndpoint == "" {
		localstackEndpoint = "http://localstack:4566"
	}

	// Setup blob storage
	var blobClient blob.Client
	var err error
	if useLocalstack {
		blobClient, err = blob.NewLocalstackClient(localstackEndpoint, azureAccountName, azureAccountKey, azureContainerName)
	} else {
		blobClient, err = blob.NewAzureClient(azureAccountName, azureAccountKey, azureContainerName)
	}
	if err != nil {
		slog.Error("blob client init failed", "error", err)
		os.Exit(1)
	}

	// Setup generator (calls Python assignment-gen)
	gen := generator.NewPythonGenerator("assignment-gen")

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Connect to RabbitMQ
	conn, err := amqp.Dial(rabbitURL)
	if err != nil {
		slog.Error("amqp dial failed", "error", err)
		os.Exit(1)
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		slog.Error("amqp channel failed", "error", err)
		os.Exit(1)
	}
	defer ch.Close()

	// Declare exchange and queue
	if err := ch.ExchangeDeclare("cp.jobs", "topic", true, false, false, false, nil); err != nil {
		slog.Error("declare jobs exchange failed", "error", err)
		os.Exit(1)
	}
	if err := ch.ExchangeDeclare(resultsExchange, "topic", true, false, false, false, nil); err != nil {
		slog.Error("declare results exchange failed", "error", err)
		os.Exit(1)
	}

	// Declare queue for all assignment-gen related jobs
	q, err := ch.QueueDeclare("assignment-gen.jobs", true, false, false, false, nil)
	if err != nil {
		slog.Error("declare queue failed", "error", err)
		os.Exit(1)
	}
	// Bind to all job types we handle
	for _, jobType := range []string{"assignment-gen", "outline-gen", "outline-edit"} {
		if err := ch.QueueBind(q.Name, "job."+jobType, "cp.jobs", false, nil); err != nil {
			slog.Error("bind queue failed", "error", err, "job_type", jobType)
			os.Exit(1)
		}
	}

	deliveries, err := ch.Consume(q.Name, "assignment-gen-worker", false, false, false, false, nil)
	if err != nil {
		slog.Error("consume failed", "error", err)
		os.Exit(1)
	}

	slog.Info("assignment-gen worker started", "queue", q.Name)

	for {
		select {
		case <-ctx.Done():
			slog.Info("worker shutting down")
			return
		case d, ok := <-deliveries:
			if !ok {
				slog.Warn("deliveries closed, reconnecting...")
				time.Sleep(5 * time.Second)
				// Reconnect logic would go here
				return
			}
			go handleJob(ctx, ch, d, gen, blobClient, resultsExchange)
		}
	}
}

type JobCommand struct {
	JobID         string          `json:"job_id"`
	Type          string          `json:"type"`
	UserID        string          `json:"user_id"`
	Payload       json.RawMessage `json:"payload"`
	CorrelationID string          `json:"correlation_id"`
}

func handleJob(ctx context.Context, ch *amqp.Channel, d amqp.Delivery, gen generator.Generator, blobClient blob.Client, resultsExchange string) {
	var cmd JobCommand
	if err := json.Unmarshal(d.Body, &cmd); err != nil {
		slog.Error("unmarshal job command failed", "error", err, "body", string(d.Body))
		_ = d.Nack(false, false)
		return
	}

	slog.Info("processing job", "job_id", cmd.JobID, "type", cmd.Type)

	var resultJSON []byte

	switch cmd.Type {
	case "outline-gen":
		// Generate outline only
		outline, err := gen.GenerateOutline(ctx, cmd.Payload)
		if err != nil {
			slog.Error("outline generation failed", "job_id", cmd.JobID, "error", err)
			errMsg := err.Error()
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}
		resultJSON, err := json.Marshal(map[string]interface{}{"outline": outline})
		if err != nil {
			slog.Error("marshal outline result failed", "job_id", cmd.JobID, "error", err)
			errMsg := fmt.Sprintf("marshal result failed: %v", err)
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}
		publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "COMPLETED", resultJSON, nil)
		_ = d.Ack(false)
		slog.Info("outline generation completed", "job_id", cmd.JobID)

	case "outline-edit":
		// Edit outline
		var payloadMap map[string]interface{}
		if err := json.Unmarshal(cmd.Payload, &payloadMap); err != nil {
			slog.Error("unmarshal payload failed", "job_id", cmd.JobID, "error", err)
			errMsg := fmt.Sprintf("invalid payload: %v", err)
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}

		outlineRaw, _ := json.Marshal(payloadMap["outline"])
		userMessage, _ := payloadMap["user_message"].(string)

		result, err := gen.EditOutline(ctx, outlineRaw, userMessage)
		if err != nil {
			slog.Error("outline editing failed", "job_id", cmd.JobID, "error", err)
			errMsg := err.Error()
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}
		resultJSON, err := json.Marshal(result)
		if err != nil {
			slog.Error("marshal edit result failed", "job_id", cmd.JobID, "error", err)
			errMsg := fmt.Sprintf("marshal result failed: %v", err)
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}
		publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "COMPLETED", resultJSON, nil)
		_ = d.Ack(false)
		slog.Info("outline editing completed", "job_id", cmd.JobID)

	case "assignment-gen":
		// Generate full assignment document
		result, err := gen.Generate(ctx, cmd.Payload)
		if err != nil {
			slog.Error("generation failed", "job_id", cmd.JobID, "error", err)
			errMsg := err.Error()
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}

		// Verify file exists before upload
		if _, err := os.Stat(result.DocPath); os.IsNotExist(err) {
			slog.Error("generated file does not exist", "job_id", cmd.JobID, "path", result.DocPath)
			errMsg := fmt.Sprintf("generated file not found: %s", result.DocPath)
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}

		slog.Info("file verified, uploading to blob storage", "job_id", cmd.JobID, "path", result.DocPath)

		// Upload to blob storage
		downloadURL, err := blobClient.Upload(ctx, cmd.JobID, result.DocPath)
		
		// Clean up temp directory after upload (whether success or failure)
		if result.DocPath != "" {
			tmpDir := filepath.Dir(result.DocPath)
			if tmpDir != "" && strings.Contains(tmpDir, "assignment-gen-") {
				if err := os.RemoveAll(tmpDir); err != nil {
					slog.Warn("failed to cleanup temp dir", "dir", tmpDir, "error", err)
				} else {
					slog.Info("cleaned up temp directory", "dir", tmpDir)
				}
			}
		}
		
		if err != nil {
			slog.Error("upload failed", "job_id", cmd.JobID, "error", err)
			errMsg := fmt.Sprintf("upload failed: %v", err)
			publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
			_ = d.Ack(false)
			return
		}

		// Publish success result
		resultJSON, _ = json.Marshal(map[string]string{"download_url": downloadURL})
		publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "COMPLETED", resultJSON, nil)
		_ = d.Ack(false)
		slog.Info("job completed", "job_id", cmd.JobID, "download_url", downloadURL)

	default:
		slog.Error("unknown job type", "job_id", cmd.JobID, "type", cmd.Type)
		errMsg := fmt.Sprintf("unknown job type: %s", cmd.Type)
		publishResult(ch, resultsExchange, cmd.JobID, cmd.Type, cmd.CorrelationID, "FAILED", nil, &errMsg)
		_ = d.Ack(false)
		return
	}
}

func publishResult(ch *amqp.Channel, exchange, jobID, jobType, correlationID, status string, result []byte, errMsg *string) {
	// Use job type for routing key
	routingKey := "result." + jobType
	event := map[string]interface{}{
		"job_id":         jobID,
		"type":            jobType,
		"status":          status,
		"correlation_id":  correlationID,
	}
	if len(result) > 0 {
		// Unmarshal result to preserve JSON structure (not base64 encode it)
		var resultObj interface{}
		if err := json.Unmarshal(result, &resultObj); err == nil {
			event["result"] = resultObj
		} else {
			// Fallback: if unmarshal fails, use raw message (shouldn't happen)
			event["result"] = json.RawMessage(result)
		}
	}
	if errMsg != nil {
		event["error"] = *errMsg
	}
	body, err := json.Marshal(event)
	if err != nil {
		slog.Error("marshal result event failed", "job_id", jobID, "error", err)
		return
	}
	err = ch.PublishWithContext(context.Background(), exchange, routingKey, false, false, amqp.Publishing{
		ContentType:  "application/json",
		Body:         body,
		DeliveryMode: amqp.Persistent,
	})
	if err != nil {
		slog.Error("publish result failed", "job_id", jobID, "status", status, "error", err)
	} else {
		slog.Info("published result", "job_id", jobID, "status", status)
	}
}
