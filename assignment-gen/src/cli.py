"""
Project: Studojo Agents
File: src/cli.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 17 November 2025
Description: CLI formatting utilities for better user experience.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models.assignment import AssignmentSpec


def format_outline(outline: Dict[str, Any]) -> str:
    """
    Format an assignment outline as a human-readable string.

    Args:
        outline: Dictionary with "sections" key containing list of sections

    Returns:
        Formatted string suitable for display in CLI/TUI
    """
    sections = outline.get("sections", [])
    if not sections:
        return "No outline sections found."

    lines = []
    lines.append("=" * 70)
    lines.append("ASSIGNMENT OUTLINE")
    lines.append("=" * 70)
    lines.append("")

    total_words = sum(s.get("target_words", 0) for s in sections)

    for section in sections:
        section_id = str(section.get("id", ""))
        section_title = str(section.get("title", ""))
        section_words = int(section.get("target_words", 0))
        subsections = section.get("subsections", [])
        notes = section.get("notes", "")

        # Section header
        lines.append(f"Section {section_id}: {section_title}")
        lines.append(f"  Target: ~{section_words:,} words")
        if notes:
            lines.append(f"  Note: {notes}")
        lines.append("")

        # Subsections
        if subsections:
            for sub in subsections:
                sub_id = str(sub.get("id", ""))
                sub_title = str(sub.get("title", ""))
                sub_words = int(sub.get("target_words", 0))
                lines.append(f"  {sub_id}  {sub_title} (~{sub_words:,} words)")
            lines.append("")
        else:
            lines.append("  (No subsections)")
            lines.append("")

    lines.append("-" * 70)
    lines.append(f"Total target: ~{total_words:,} words")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_spec_summary(spec: AssignmentSpec) -> str:
    """
    Format an assignment spec as a human-readable summary.

    Args:
        spec: AssignmentSpec instance

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ASSIGNMENT SPECIFICATION")
    lines.append("=" * 70)
    lines.append("")

    # Core info
    if spec.assignment_type:
        lines.append(f"Type: {spec.assignment_type}")
    if spec.course_name:
        lines.append(f"Course: {spec.course_name}")
    if spec.level:
        lines.append(f"Level: {spec.level}")
    lines.append("")

    # Description
    if spec.description:
        desc = spec.description[:200] + "..." if len(spec.description) > 200 else spec.description
        lines.append(f"Description: {desc}")
        lines.append("")

    # Length
    length_parts = []
    if spec.length_pages:
        length_parts.append(f"{spec.length_pages} pages")
    if spec.length_words:
        length_parts.append(f"{spec.length_words:,} words")
    if spec.length_notes:
        length_parts.append(spec.length_notes)
    if length_parts:
        lines.append(f"Length: {', '.join(length_parts)}")
        lines.append("")

    # Formatting
    if spec.format_style:
        lines.append(f"Format: {spec.format_style}")
    if spec.tone:
        lines.append(f"Tone: {spec.tone}")
    lines.append("")

    # Sources and features
    if spec.allow_web_search is not None:
        lines.append(f"Web Search: {'Enabled' if spec.allow_web_search else 'Disabled'}")
    if spec.allows_graphs is not None:
        lines.append(f"Graphs/Tables: {'Enabled' if spec.allows_graphs else 'Disabled'}")
    if spec.graph_requirements:
        lines.append(f"Graph Requirements: {', '.join(spec.graph_requirements)}")
    if spec.citation_sources_policy:
        lines.append(f"Citation Policy: {spec.citation_sources_policy}")
    lines.append("")

    # Structure
    if spec.structure_requirements:
        lines.append(f"Required Sections: {', '.join(spec.structure_requirements)}")
    if spec.forbidden_sections:
        lines.append(f"Forbidden Sections: {', '.join(spec.forbidden_sections)}")
    lines.append("")

    # Humanizer
    if spec.humanizer_config.enabled:
        lines.append(f"Humanization: Enabled ({spec.humanizer_config.intensity} intensity)")
    else:
        lines.append("Humanization: Disabled")
    lines.append("")

    # Status
    lines.append(f"Status: {spec.status.value.upper()}")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_section_header(title: str, level: int = 1) -> str:
    """
    Format a section header for display.

    Args:
        title: Section title
        level: Header level (1-3)

    Returns:
        Formatted header string
    """
    if level == 1:
        return f"\n{'=' * 70}\n{title.upper()}\n{'=' * 70}\n"
    elif level == 2:
        return f"\n{'-' * 70}\n{title}\n{'-' * 70}\n"
    else:
        return f"\n{title}\n{'-' * 40}\n"


def format_table(data: List[Dict[str, Any]], headers: List[str] | None = None) -> str:
    """
    Format tabular data nicely.

    Args:
        data: List of dictionaries with data rows
        headers: Optional list of header names (uses dict keys if not provided)

    Returns:
        Formatted table string
    """
    if not data:
        return "No data to display."

    if headers is None:
        headers = list(data[0].keys()) if data else []

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], len(value))

    # Build table
    lines = []
    separator = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_row = "| " + " | ".join(str(h).ljust(col_widths[h]) for h in headers) + " |"

    lines.append(separator)
    lines.append(header_row)
    lines.append(separator)

    for row in data:
        data_row = "| " + " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers) + " |"
        lines.append(data_row)

    lines.append(separator)

    return "\n".join(lines)

