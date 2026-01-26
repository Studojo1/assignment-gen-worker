package generator

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/studojo/assignment-gen-worker/internal/generator/types"
)

// Generator generates assignments from payloads.
type Generator interface {
	Generate(ctx context.Context, payload []byte) (*Result, error)
	GenerateOutline(ctx context.Context, payload []byte) (map[string]interface{}, error)
	EditOutline(ctx context.Context, outline json.RawMessage, userMessage string) (map[string]interface{}, error)
}

// Result contains the generated document path.
type Result struct {
	DocPath string
	Outline map[string]interface{} // For outline-only results
}

// PythonGenerator calls the Python assignment-gen service.
type PythonGenerator struct {
	pythonPath string
	scriptPath string
}

// NewPythonGenerator creates a generator that calls Python assignment-gen.
func NewPythonGenerator(serviceName string) *PythonGenerator {
	return &PythonGenerator{
		pythonPath: os.Getenv("PYTHON_PATH"),
		scriptPath: os.Getenv("ASSIGNMENT_GEN_SCRIPT_PATH"),
	}
}

// GenerateOutline generates only the outline and returns it as JSON.
func (g *PythonGenerator) GenerateOutline(ctx context.Context, payload []byte) (map[string]interface{}, error) {
	var p types.Payload
	if err := json.Unmarshal(payload, &p); err != nil {
		return nil, fmt.Errorf("unmarshal payload: %w", err)
	}

	// Create temp directory for output
	tmpDir, err := os.MkdirTemp("", "assignment-gen-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputPath := filepath.Join(tmpDir, "outline.json")

	// Write payload to temp file for Python script
	payloadFile := filepath.Join(tmpDir, "payload.json")
	if err := os.WriteFile(payloadFile, payload, 0644); err != nil {
		return nil, fmt.Errorf("write payload: %w", err)
	}

	scriptPath := "/app/assignment-gen/worker_generate_outline.py"
	if g.scriptPath != "" {
		// Use base path and replace filename
		scriptPath = filepath.Join(filepath.Dir(g.scriptPath), "worker_generate_outline.py")
	}

	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("python script not found: %s", scriptPath)
	}

	pythonCmd := g.pythonPath
	if pythonCmd == "" {
		pythonCmd = "python3"
	}

	if _, err := exec.LookPath(pythonCmd); err != nil {
		return nil, fmt.Errorf("python command not found: %s", pythonCmd)
	}

	slog.Info("starting python outline generator", "python", pythonCmd, "script", scriptPath, "payload", payloadFile, "output", outputPath)
	cmd := exec.CommandContext(ctx, pythonCmd, scriptPath, payloadFile, outputPath)
	cmd.Env = append(os.Environ(),
		"OPENAI_API_KEY="+os.Getenv("OPENAI_API_KEY"),
		"ANTHROPIC_API_KEY="+os.Getenv("ANTHROPIC_API_KEY"),
		"REPHRASY_API_KEY="+os.Getenv("REPHRASY_API_KEY"),
		"PYTHONPATH=/app/assignment-gen/src:/app/assignment-gen",
		"DOC_OUTPUT_DIR="+tmpDir,
	)
	cmd.Dir = "/app/assignment-gen"

	output, err := cmd.CombinedOutput()
	if len(output) > 0 {
		slog.Info("python outline generator output", "output", string(output))
	}
	if err != nil {
		slog.Error("python outline generator failed", "error", err, "output", string(output))
		return nil, fmt.Errorf("python outline generator: %w", err)
	}
	slog.Info("python outline generator completed successfully")

	// Read outline from output file
	outlineData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read outline output: %w", err)
	}

	var outline map[string]interface{}
	if err := json.Unmarshal(outlineData, &outline); err != nil {
		return nil, fmt.Errorf("unmarshal outline: %w", err)
	}

	return outline, nil
}

// EditOutline edits an outline based on user message.
func (g *PythonGenerator) EditOutline(ctx context.Context, outline json.RawMessage, userMessage string) (map[string]interface{}, error) {
	// Create temp directory for output
	tmpDir, err := os.MkdirTemp("", "assignment-gen-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputPath := filepath.Join(tmpDir, "outline.json")

	// Create payload with outline and user message
	payload := map[string]interface{}{
		"outline":      json.RawMessage(outline),
		"user_message": userMessage,
	}
	payloadBytes, _ := json.Marshal(payload)

	// Write payload to temp file for Python script
	payloadFile := filepath.Join(tmpDir, "payload.json")
	if err := os.WriteFile(payloadFile, payloadBytes, 0644); err != nil {
		return nil, fmt.Errorf("write payload: %w", err)
	}

	scriptPath := "/app/assignment-gen/worker_edit_outline.py"
	if g.scriptPath != "" {
		scriptPath = filepath.Join(filepath.Dir(g.scriptPath), "worker_edit_outline.py")
	}

	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("python script not found: %s", scriptPath)
	}

	pythonCmd := g.pythonPath
	if pythonCmd == "" {
		pythonCmd = "python3"
	}

	if _, err := exec.LookPath(pythonCmd); err != nil {
		return nil, fmt.Errorf("python command not found: %s", pythonCmd)
	}

	slog.Info("starting python outline editor", "python", pythonCmd, "script", scriptPath, "payload", payloadFile, "output", outputPath)
	cmd := exec.CommandContext(ctx, pythonCmd, scriptPath, payloadFile, outputPath)
	cmd.Env = append(os.Environ(),
		"OPENAI_API_KEY="+os.Getenv("OPENAI_API_KEY"),
		"ANTHROPIC_API_KEY="+os.Getenv("ANTHROPIC_API_KEY"),
		"REPHRASY_API_KEY="+os.Getenv("REPHRASY_API_KEY"),
		"PYTHONPATH=/app/assignment-gen/src:/app/assignment-gen",
		"DOC_OUTPUT_DIR="+tmpDir,
	)
	cmd.Dir = "/app/assignment-gen"

	output, err := cmd.CombinedOutput()
	if len(output) > 0 {
		slog.Info("python outline editor output", "output", string(output))
	}
	if err != nil {
		slog.Error("python outline editor failed", "error", err, "output", string(output))
		return nil, fmt.Errorf("python outline editor: %w", err)
	}
	slog.Info("python outline editor completed successfully")

	// Read result from output file
	resultData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read outline output: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resultData, &result); err != nil {
		return nil, fmt.Errorf("unmarshal result: %w", err)
	}

	return result, nil
}

// Generate runs the Python assignment generator and returns the docx path.
func (g *PythonGenerator) Generate(ctx context.Context, payload []byte) (*Result, error) {
	var p types.Payload
	if err := json.Unmarshal(payload, &p); err != nil {
		return nil, fmt.Errorf("unmarshal payload: %w", err)
	}

	// Create temp directory for output
	tmpDir, err := os.MkdirTemp("", "assignment-gen-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	// Don't defer cleanup here - the file needs to persist until after blob upload
	// Cleanup will happen in the worker after successful upload

	outputPath := filepath.Join(tmpDir, "assignment.docx")

	// Write payload to temp file for Python script
	payloadFile := filepath.Join(tmpDir, "payload.json")
	if err := os.WriteFile(payloadFile, payload, 0644); err != nil {
		return nil, fmt.Errorf("write payload: %w", err)
	}

	// Call Python script in assignment-gen service
	// In Docker Compose, we share the assignment-gen code via volume
	scriptPath := g.scriptPath
	if scriptPath == "" {
		scriptPath = "/app/assignment-gen/worker_generate.py"
	}

	// Verify script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("python script not found: %s", scriptPath)
	}

	pythonCmd := g.pythonPath
	if pythonCmd == "" {
		pythonCmd = "python3"
	}

	// Verify Python command exists
	if _, err := exec.LookPath(pythonCmd); err != nil {
		return nil, fmt.Errorf("python command not found: %s (PATH: %s)", pythonCmd, os.Getenv("PATH"))
	}

	// Run Python generator
	// Set DOC_OUTPUT_DIR to the writable temp directory so Python can write generated files there
	slog.Info("starting python generator", "python", pythonCmd, "script", scriptPath, "payload", payloadFile, "output", outputPath, "workdir", "/app/assignment-gen", "doc_output_dir", tmpDir)
	cmd := exec.CommandContext(ctx, pythonCmd, scriptPath, payloadFile, outputPath)
	cmd.Env = append(os.Environ(),
		"OPENAI_API_KEY="+os.Getenv("OPENAI_API_KEY"),
		"ANTHROPIC_API_KEY="+os.Getenv("ANTHROPIC_API_KEY"),
		"REPHRASY_API_KEY="+os.Getenv("REPHRASY_API_KEY"),
		"PYTHONPATH=/app/assignment-gen/src:/app/assignment-gen",
		"DOC_OUTPUT_DIR="+tmpDir,
	)
	cmd.Dir = "/app/assignment-gen"

	// Capture both stdout and stderr
	output, err := cmd.CombinedOutput()
	if len(output) > 0 {
		slog.Info("python generator output", "output", string(output))
	}
	if err != nil {
		slog.Error("python generator failed", "error", err, "output", string(output))
		return nil, fmt.Errorf("python generator: %w", err)
	}
	slog.Info("python generator completed successfully")

	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("output file not created: %s", outputPath)
	}

	// Copy to a persistent location (or return temp path - blob upload will handle it)
	// For now, return the temp path - the blob client will upload and we can clean up
	return &Result{DocPath: outputPath}, nil
}
