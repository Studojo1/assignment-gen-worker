"""
Assignment-related chains module.

This module contains all chains related to assignment generation:
- assignment_input: Input collection chain
- assignment_outline: Outline generation chain
- assignment_content: Content generation chain
- assignment_pipeline: High-level pipeline chain
"""

from .assignment_content import (
    SubsectionSpec,
    build_assignment_content_chain,
)
from .assignment_input import (
    AssignmentInputCollectorOutput,
    build_assignment_input_collector_chain,
)
from .assignment_outline import build_assignment_outline_chain
from .assignment_outline_editor import (
    OutlineEditorOutput,
    build_assignment_outline_editor_chain,
)
from .assignment_pipeline import build_assignment_pipeline

__all__ = [
    "SubsectionSpec",
    "build_assignment_content_chain",
    "AssignmentInputCollectorOutput",
    "build_assignment_input_collector_chain",
    "build_assignment_outline_chain",
    "build_assignment_outline_editor_chain",
    "OutlineEditorOutput",
    "build_assignment_pipeline",
]
