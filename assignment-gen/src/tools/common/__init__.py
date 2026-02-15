"""
Common/shared tools used across different features.
"""

from tools.common.doc import generate_formatted_document
from tools.common.graph import (
    analyze_content_for_graphs,
    generate_content_aware_graph,
    generate_matplotlib_graph,
    set_agent_llm_for_graph,
)
from tools.common.math import (
    parse_equations,
    render_block_equation,
    render_inline_equation,
    sanitize_latex,
)
from tools.common.rephrasy import RephrasyService
from tools.common.summarize_file import (
    read_file_content,
    set_agent_llm_for_summarize,
    summarize_file,
)
from tools.common.uniqueness import UniquenessChecker, UniquenessResult
from tools.common.web import (
    read_webpage_content,
    search_and_fetch,
    search_duckduckgo,
)

__all__ = [
    "generate_formatted_document",
    "analyze_content_for_graphs",
    "generate_content_aware_graph",
    "generate_matplotlib_graph",
    "set_agent_llm_for_graph",
    "parse_equations",
    "render_block_equation",
    "render_inline_equation",
    "sanitize_latex",
    "RephrasyService",
    "read_file_content",
    "set_agent_llm_for_summarize",
    "summarize_file",
    "UniquenessChecker",
    "UniquenessResult",
    "read_webpage_content",
    "search_and_fetch",
    "search_duckduckgo",
]

