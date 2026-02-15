"""
Project: Studojo Agents
File: src/tools/math.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 17 November 2025
Description: Equation parsing and rendering utilities for DOCX documents.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import math2docx
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

logger = logging.getLogger(__name__)


def parse_equations(content: str) -> List[Dict[str, Any]]:
    """
    Parse inline and block equations from content string.

    Detects:
    - Inline equations: $...$ (e.g., $x^2 + y^2 = r^2$)
    - Block equations: $$...$$ (e.g., $$\\int_0^1 x dx$$)

    Args:
        content: String containing text with LaTeX equations

    Returns:
        List of equation dicts with keys:
        - type: "inline" or "block"
        - latex: LaTeX content (without delimiters, unescaped)
        - start_pos: Start position in original text
        - end_pos: End position in original text
    """
    equations = []
    pos = 0

    def unescape_latex(latex: str) -> str:
        """
        Unescape LaTeX that may have been double-escaped in JSON.
        Converts \\begin to \begin, \\\\ to \\, etc.
        """
        # Replace double backslashes with single, but preserve \\\\ as \\
        # This handles cases where JSON escaping added extra backslashes
        # We need to be careful: \\\\ should become \\ (two backslashes)
        # But \\begin should become \begin (one backslash)
        # Strategy: replace \\\\ first (to preserve intentional double backslashes),
        # then replace remaining \\ with \
        # Use a safe placeholder that won't appear in LaTeX (not NULL bytes)
        placeholder = "__DOUBLE_BACKSLASH_PLACEHOLDER__"
        result = latex.replace("\\\\\\\\", placeholder)  # Preserve intentional doubles
        result = result.replace("\\\\", "\\")  # Replace double with single
        result = result.replace(placeholder, "\\\\")  # Restore intentional doubles
        return result

    while pos < len(content):
        # Look for block equations first ($$...$$)
        block_match = re.search(r"\$\$(.+?)\$\$", content[pos:], re.DOTALL)
        if block_match:
            start = pos + block_match.start()
            end = pos + block_match.end()
            latex = unescape_latex(block_match.group(1).strip())
            equations.append({
                "type": "block",
                "latex": latex,
                "start_pos": start,
                "end_pos": end,
            })
            pos = end
            continue

        # Look for inline equations ($...$)
        # Avoid matching $$ as inline
        inline_match = re.search(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", content[pos:], re.DOTALL)
        if inline_match:
            start = pos + inline_match.start()
            end = pos + inline_match.end()
            latex = unescape_latex(inline_match.group(1).strip())
            equations.append({
                "type": "inline",
                "latex": latex,
                "start_pos": start,
                "end_pos": end,
            })
            pos = end
            continue

        # No more equations found
        break

    return equations


def sanitize_latex(latex: str) -> str:
    """
    Sanitize LaTeX string to remove NULL bytes and invalid XML characters.
    """
    if not latex:
        return latex
    
    result = []
    for char in latex:
        code = ord(char)
        # Allow valid XML characters (same as sanitize_xml_text)
        if code == 0x09 or code == 0x0A or code == 0x0D:  # Tab, LF, CR
            result.append(char)
        elif 0x20 <= code <= 0xD7FF:
            result.append(char)
        elif 0xE000 <= code <= 0xFFFD:
            result.append(char)
        elif code == 0x00:
            continue  # Skip NULL bytes
        elif code < 0x20:
            continue  # Skip other control characters
        else:
            result.append(char)
    
    return "".join(result)


def render_equation(
    doc: Document,
    paragraph: Any,
    latex: str,
    inline: bool = True,
    font_size: int = 12,
) -> None:
    """
    Render a LaTeX equation into a paragraph.

    Args:
        doc: Document object (for fallback)
        paragraph: Paragraph object to add equation to
        latex: LaTeX equation string
        inline: Whether equation is inline (True) or block/display (False)
        font_size: Font size for fallback text
    """
    # Sanitize LaTeX to remove NULL bytes and invalid characters
    latex = sanitize_latex(latex)
    
    try:
        # Use math2docx to render the equation
        math2docx.add_math(paragraph, latex)
    except Exception as e:
        logger.warning(f"Failed to render equation '{latex}': {e}. Using plain text fallback.")
        # Fallback: render as plain text (sanitize again for safety)
        fallback_text = f"[${latex}$]" if inline else f"[$${latex}$$]"
        fallback_text = sanitize_latex(fallback_text)
        run = paragraph.add_run(fallback_text)
        run.font.size = font_size


def render_block_equation(doc: Document, latex: str) -> None:
    """
    Render a block/display equation (centered, with spacing).

    Args:
        doc: Document object
        latex: LaTeX equation string
    """
    # Sanitize LaTeX to remove NULL bytes and invalid characters
    latex = sanitize_latex(latex)
    
    # Create centered paragraph for block equation
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    try:
        math2docx.add_math(paragraph, latex)
    except Exception as e:
        logger.warning(f"Failed to render block equation '{latex}': {e}. Using plain text fallback.")
        # Sanitize fallback text
        fallback_text = sanitize_latex(f"[$${latex}$$]")
        run = paragraph.add_run(fallback_text)
        run.font.size = 12

    # Add blank paragraph after for spacing
    doc.add_paragraph()


def render_inline_equation(
    paragraph: Any,
    run: Any,
    latex: str,
    font_size: int = 12,
) -> None:
    """
    Render an inline equation within a paragraph.

    Args:
        paragraph: Paragraph object
        run: Current run object (will create new run for equation)
        latex: LaTeX equation string
        font_size: Font size for fallback text
    """
    # Sanitize LaTeX to remove NULL bytes and invalid characters
    latex = sanitize_latex(latex)
    
    try:
        # Add space before equation if there's preceding text
        if run.text and run.text.rstrip():
            run.add_text(" ")

        # Render equation inline
        math2docx.add_math(paragraph, latex)

        # Add space after equation for next text
        new_run = paragraph.add_run(" ")
        new_run.font.size = font_size
    except Exception as e:
        logger.warning(f"Failed to render inline equation '{latex}': {e}. Using plain text fallback.")
        # Sanitize fallback text
        fallback_text = sanitize_latex(f" [${latex}$] ")
        run.add_text(fallback_text)

