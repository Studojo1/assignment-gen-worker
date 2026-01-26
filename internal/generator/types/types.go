package types

import "encoding/json"

// Payload matches control-plane's assignmentgen.Payload.
type Payload struct {
	AssignmentType  string          `json:"assignment_type"`
	Description     string          `json:"description"`
	LengthWords     *int            `json:"length_words,omitempty"`
	LengthPages     *float64         `json:"length_pages,omitempty"`
	FormatStyle     string          `json:"format_style"`
	CourseName      string          `json:"course_name,omitempty"`
	AllowWebSearch  bool            `json:"allow_web_search"`
	Outline         *json.RawMessage `json:"outline,omitempty"` // Pre-generated outline for final generation
	HumanizerConfig *struct {
		Enabled   bool   `json:"enabled"`
		Intensity string `json:"intensity"`
	} `json:"humanizer_config,omitempty"`
}
