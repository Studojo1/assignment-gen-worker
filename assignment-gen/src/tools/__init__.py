"""
Project: Studojo Assignment Generator
File: src/tools/__init__.py
Description: Export all available tools for assignment generation.
"""

# Import from submodules
from tools.assignment import (
    collect_assignment_input,
    generate_assignment,
    set_agent_llm,
    # New operation-level tools
    generate_section_content,
    generate_section_graph,
    humanize_section_content,
    build_assignment_document,
    check_content_uniqueness,
    generate_assignment_outline,
    get_assignment_spec,
    update_section_content,
)
from tools.common import (
    generate_formatted_document,
    generate_content_aware_graph,
    generate_matplotlib_graph,
    read_file_content,
    summarize_file,
    search_duckduckgo,
    read_webpage_content,
)

__all__ = [
    # Assignment tools (existing)
    "collect_assignment_input",
    "generate_assignment",
    "set_agent_llm",
    # Assignment tools (new operation-level)
    "generate_section_content",
    "generate_section_graph",
    "humanize_section_content",
    "build_assignment_document",
    "check_content_uniqueness",
    "generate_assignment_outline",
    "get_assignment_spec",
    "update_section_content",
    # Document tools
    "generate_formatted_document",
    # File tools
    "summarize_file",
    "read_file_content",
    # Graph tools
    "generate_content_aware_graph",
    "generate_matplotlib_graph",
    # Web tools
    "search_duckduckgo",
    "read_webpage_content",
]
