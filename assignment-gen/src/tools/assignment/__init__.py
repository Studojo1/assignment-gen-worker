"""
Assignment-specific tools.
"""

from tools.assignment.generate_assignment import (
    collect_assignment_input,
    generate_assignment,
    set_agent_llm,
)
from tools.assignment.generate_section_content import generate_section_content
from tools.assignment.generate_section_graph import generate_section_graph
from tools.assignment.humanize_section_content import humanize_section_content
from tools.assignment.build_assignment_document import build_assignment_document
from tools.assignment.check_content_uniqueness import check_content_uniqueness
from tools.assignment.generate_assignment_outline import generate_assignment_outline
from tools.assignment.get_assignment_spec import get_assignment_spec
from tools.assignment.update_section_content import update_section_content

__all__ = [
    # Existing tools
    "collect_assignment_input",
    "generate_assignment",
    "set_agent_llm",
    # New operation-level tools
    "generate_section_content",
    "generate_section_graph",
    "humanize_section_content",
    "build_assignment_document",
    "check_content_uniqueness",
    "generate_assignment_outline",
    "get_assignment_spec",
    "update_section_content",
]


