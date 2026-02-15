"""
Project: Studojo Agents
File: src/models/assignment.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 16 November 2025
Description: Core data models for assignment handling and orchestration.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field


class AssignmentStatus(str, Enum):
    """Where we are in the assignment lifecycle from the agent's perspective."""

    COLLECTING = "collecting"  # Input collector is still asking questions
    READY = "ready"  # Spec is complete enough to generate outline/content


class HumanizerConfig(BaseModel):
    """Configuration for post-processing / humanization (e.g., via Rephrasy)."""

    enabled: bool = Field(
        default=True,
        description="Whether to run the generated text through the humanizer.",
    )
    intensity: str = Field(
        default="light",
        description="Rough level of humanization: light | medium | heavy.",
    )


class AssignmentSpec(BaseModel):
    """
    Structured specification of an assignment.

    The input collector gradually fills this in over multiple turns until
    `status` becomes READY.
    """

    status: AssignmentStatus = Field(
        default=AssignmentStatus.COLLECTING,
        description="Whether the spec is still being collected or ready to use.",
    )

    assignment_id: Optional[str] = Field(
        default=None,
        description="Internal ID if the assignment is persisted.",
    )

    # Core identity / context
    assignment_type: Optional[str] = Field(
        default=None,
        description="Type of assignment (essay, lab report, project report, etc.).",
    )
    course_name: Optional[str] = None
    level: Optional[str] = Field(
        default=None,
        description="Academic level (high school, undergrad, grad, etc.).",
    )

    # Prompt / description
    description: Optional[str] = Field(
        default=None,
        description="The user-provided assignment prompt / description.",
    )

    # Time & scope
    due_date: Optional[datetime] = Field(
        default=None,
        description="Due date/time, if known.",
    )
    length_words: Optional[int] = Field(
        default=None,
        description="Target length in words, if specified.",
    )
    length_pages: Optional[float] = Field(
        default=None,
        description="Target length in pages, if specified.",
    )
    length_notes: Optional[str] = Field(
        default=None,
        description="Any free-form description of scope (e.g., '8–10 pages').",
    )

    # Formatting & structure
    format_style: Optional[str] = Field(
        default=None,
        description="Formatting system (APA, MLA, IEEE, etc.).",
    )
    structure_requirements: List[str] = Field(
        default_factory=list,
        description="Required sections (Introduction, Literature Review, etc.).",
    )
    forbidden_sections: List[str] = Field(
        default_factory=list,
        description="Sections the user explicitly does not want.",
    )

    # Sources, web, graphs & tone
    sources_policy: Optional[str] = Field(
        default=None,
        description="How sources should be handled (web-only, user-provided-only, etc.).",
    )
    allow_web_search: Optional[bool] = Field(
        default=None,
        description="Whether the agent is allowed to use web search for this assignment.",
    )
    web_search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Web search results found during input collection, stored for use in content generation.",
    )
    allows_graphs: Optional[bool] = Field(
        default=True,
        description="Whether graphs/tables/figures are allowed or expected.",
    )
    graph_requirements: List[str] = Field(
        default_factory=list,
        description="Any specific requirements for graphs/tables/figures.",
    )
    citation_sources_policy: Optional[str] = Field(
        default=None,
        description="Constraints on which sources to cite (any, course-only, etc.).",
    )
    tone: Optional[str] = Field(
        default=None,
        description="Desired writing tone (formal academic, semi-formal, etc.).",
    )
    humanizer_config: HumanizerConfig = Field(
        default_factory=HumanizerConfig,
        description="Configuration for humanization of the final text.",
    )
    plagiarism_sensitivity: Optional[str] = Field(
        default=None,
        description="How paranoid we should be about similarity / originality.",
    )

    # Uploaded files
    uploaded_files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of uploaded files with their extracted content.",
    )
