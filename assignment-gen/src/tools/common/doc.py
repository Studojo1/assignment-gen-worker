"""
Project: Studojo Agents
File: src/tools/doc.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 16 November 2025
Description: Langchain Tools realted to Document Generation
"""

import logging
import os
from typing import Any, Dict, List, Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
from docx.shared import Pt, RGBColor
from langchain.tools import tool
from pydantic import BaseModel, Field

from tools.common.math import (
    parse_equations,
    render_block_equation,
    render_inline_equation,
)

logger = logging.getLogger(__name__)


class SimpleCitationFormatter:
    """Simple citation formatter for different styles."""

    def __init__(self, style: str = "APA"):
        self.style = style.upper()

    def _extract_organization_from_url(self, url: str) -> str:
        """Extract organization name from URL domain."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            netloc = parsed.netloc.replace("www.", "")
            # Get the main domain (e.g., "example.com" -> "Example")
            parts = netloc.split(".")
            if len(parts) >= 2:
                org_name = parts[-2]  # Second-to-last part (e.g., "example" from "example.com")
            else:
                org_name = parts[0] if parts else "Unknown"
            # Capitalize first letter
            return org_name.capitalize() if org_name else "Unknown"
        except Exception:
            return "Unknown"

    def format_reference(self, reference: Dict[str, Any], index: int = None) -> str:
        """Format a reference based on the citation style."""
        # Handle both 'authors' (plural) and 'author' (singular) fields
        authors = reference.get("authors") or reference.get("author", "")
        title = reference.get("title", "Unknown")
        year = reference.get("year") or reference.get("date", "")
        source = reference.get("source", "")
        url = reference.get("url", "")
        website = reference.get("website", "")
        
        # Detect if this is a web source
        is_web_source = bool(url)
        
        if self.style == "IEEE" and index:
            # IEEE format for web sources
            if is_web_source:
                author_field = authors if (authors and authors.strip() and authors != "Unknown") else (website or self._extract_organization_from_url(url) or "Unknown")
                return f'[{index}] {author_field}, "{title}," {url}, {year}.'
            else:
                return f'[{index}] {authors}, "{title}," {source}, {year}.'
        else:  # APA style (default)
            # APA format for web sources: Author/Organization. (Year). Title of page. Website Name. URL
            if is_web_source:
                # Determine author/organization
                if authors and authors.strip() and authors != "Unknown":
                    author_field = authors
                elif website and website.strip():
                    # Extract organization name from website domain
                    author_field = self._extract_organization_from_url(website) if website else "Unknown"
                else:
                    author_field = self._extract_organization_from_url(url) or "Unknown"
                
                # Handle missing year
                year_field = f"({year})" if year else "(n.d.)"
                
                # Get website name (use domain name from URL if website field not available)
                if website:
                    website_name = self._extract_organization_from_url(website)
                else:
                    website_name = self._extract_organization_from_url(url)
                
                # Format: Author. (Year). Title. Website Name. URL
                return f"{author_field}. {year_field}. {title}. {website_name}. {url}"
            else:
                # APA format for non-web sources
                author_field = authors if (authors and authors.strip()) else "Unknown"
                year_field = f"({year})" if year else "(n.d.)"
                return f"{author_field} {year_field}. {title}. {source}."


def get_formatter(style: str = "APA"):
    """Get citation formatter for the specified style."""
    return SimpleCitationFormatter(style)


def add_image_to_document(
    doc: Document,
    image_path: str,
    width: Optional[float] = None,
    caption: Optional[str] = None,
) -> None:
    """
    Add an image to a Word document.

    Args:
        doc: Document object
        image_path: Path to the image file (can be absolute or relative)
        width: Optional width in inches (default: 6 inches)
        caption: Optional caption text to add below the image
    """
    import tempfile
    from docx.shared import Inches

    # Resolve image path - ensure it's absolute
    resolved_path = image_path
    if not os.path.isabs(resolved_path):
        # Try relative to current working directory
        cwd_path = os.path.abspath(resolved_path)
        if os.path.exists(cwd_path):
            resolved_path = cwd_path
        else:
            # Try relative to document output directory
            doc_output_dir = os.getenv("DOC_OUTPUT_DIR", "generated_docs")
            doc_output_path = os.path.join(
                os.path.abspath(doc_output_dir), os.path.basename(resolved_path)
            )
            if os.path.exists(doc_output_path):
                resolved_path = doc_output_path
            else:
                # Try relative to temp directory (where graphs might be generated)
                temp_dir = os.path.abspath(tempfile.gettempdir())
                temp_path = os.path.join(temp_dir, os.path.basename(resolved_path))
                if os.path.exists(temp_path):
                    resolved_path = temp_path

    # Final check - ensure absolute path
    resolved_path = os.path.abspath(resolved_path) if resolved_path else image_path

    if not os.path.exists(resolved_path):
        logger.error(f"Image file not found: {resolved_path} (original: {image_path})")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"DOC_OUTPUT_DIR: {os.getenv('DOC_OUTPUT_DIR', 'not set')}")
        # Try to find the file by basename in common locations
        filename = os.path.basename(resolved_path)
        search_dirs = [
            os.getcwd(),
            os.getenv("DOC_OUTPUT_DIR", "generated_docs"),
            tempfile.gettempdir(),
        ]
        for search_dir in search_dirs:
            if search_dir and os.path.exists(search_dir):
                potential_path = os.path.join(os.path.abspath(search_dir), filename)
                if os.path.exists(potential_path):
                    logger.info(
                        f"Found image at alternative location: {potential_path}"
                    )
                    resolved_path = potential_path
                    break
        else:
            logger.error(f"Could not locate image file: {filename}")
            return

    logger.info(
        f"Adding image to document: {resolved_path} (exists: {os.path.exists(resolved_path)}, size: {os.path.getsize(resolved_path)} bytes)"
    )

    try:
        # Add paragraph for image (centered)
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add image
        if width is None:
            width = Inches(6)  # Default width
        else:
            width = Inches(width)

        run = para.add_run()
        run.add_picture(resolved_path, width=width)

        # Add caption if provided
        if caption:
            caption_para = doc.add_paragraph()
            caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption_para.add_run(caption)
            caption_run.font.italic = True
            caption_run.font.size = Pt(10)

        # Add spacing after image
        doc.add_paragraph()

    except Exception as e:
        logger.error(f"Error adding image to document: {e}", exc_info=True)
        logger.error(f"  Image path: {image_path}")
        logger.error(f"  Resolved path: {resolved_path}")


def parse_markdown_table(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a markdown table from text.

    Returns:
        Dict with keys:
        - 'rows': List of lists of cell contents
        - 'start_pos': Start position in text
        - 'end_pos': End position in text
        None if no table found
    """
    import re

    lines = text.split("\n")
    table_rows = []
    table_start = None
    table_end = None
    separator_found = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line looks like a table row (starts and ends with |)
        if line.startswith("|") and line.endswith("|") and len(line) > 2:
            # Check if it's a separator row (contains only dashes, colons, spaces, and pipes)
            if re.match(r"^\|[\s\-:]+\|$", line):
                # This is a separator row
                if table_start is None:
                    # Separator before any header row - invalid table, skip
                    i += 1
                    continue
                # Valid separator - mark it and continue
                separator_found = True
                i += 1
                continue

            # Parse the row (header or data row)
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if table_start is None:
                table_start = i
            table_rows.append(cells)
            table_end = i
            i += 1
        else:
            # Not a table row
            if table_rows and separator_found:
                # We had a valid table (with separator), now we're done
                break
            elif table_rows:
                # We had rows but no separator yet - might still be building table
                # Allow one blank line between header and separator
                if not line:  # Blank line
                    i += 1
                    continue
                # Check if next line might be separator
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r"^\|[\s\-:]+\|$", next_line):
                        # Next line is separator, skip this blank line and continue
                        i += 1
                        continue
                # No separator coming - but if we have 2+ rows, accept it as a table anyway
                # (some markdown tables don't have separators)
                if len(table_rows) >= 2:
                    # Accept table without separator
                    break
                # Otherwise, reset - this wasn't a table
                table_rows = []
                table_start = None
                table_end = None
                separator_found = False
            i += 1

    # Only return table if we have at least 2 rows (header + at least one data row)
    # Separator is preferred but not strictly required if we have multiple rows
    if not table_rows or len(table_rows) < 2:
        return None

    # Accept table if we have separator OR if we have 2+ rows (some markdown tables don't have separators)
    # The separator check is already handled above, so if we reach here with 2+ rows, accept it

    # Calculate positions in original text
    start_pos = sum(len(lines[j]) + 1 for j in range(table_start)) if table_start else 0
    end_pos = (
        sum(len(lines[j]) + 1 for j in range(table_end + 1)) if table_end else len(text)
    )

    return {
        "rows": table_rows,
        "start_pos": start_pos,
        "end_pos": end_pos,
    }


def parse_markdown_images(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse markdown image syntax from text.

    Supports:
    - ![alt_text](image_path)
    - ![alt_text](image_path "caption")

    Returns:
        List of dicts with keys:
        - 'alt': Alt text
        - 'path': Image path/URL
        - 'caption': Optional caption (if provided in quotes)
        - 'start_pos': Start position in text
        - 'end_pos': End position in text
        None if no images found
    """
    import re

    # Pattern to match markdown images: ![alt](path) or ![alt](path "caption")
    # Group 1: alt text
    # Group 2: path
    # Group 3: optional caption in quotes
    pattern = r'!\[([^\]]*)\]\(([^\)]+?)(?:\s+"([^"]+)")?\)'

    matches = list(re.finditer(pattern, text))

    if not matches:
        return None

    images = []
    for match in matches:
        alt_text = match.group(1) or ""
        image_path = match.group(2).strip()
        caption = match.group(3) if match.group(3) else None

        images.append(
            {
                "alt": alt_text,
                "path": image_path,
                "caption": caption or alt_text,  # Use alt text as caption fallback
                "start_pos": match.start(),
                "end_pos": match.end(),
            }
        )

    return images


def sanitize_xml_text(text: str) -> str:
    """
    Remove NULL bytes and other invalid XML characters from text.

    XML 1.0 allows:
    - #x9 (tab)
    - #xA (line feed)
    - #xD (carriage return)
    - #x20-#xD7FF
    - #xE000-#xFFFD
    - #x10000-#x10FFFF (surrogates)

    We remove NULL bytes (#x0) and other control characters that aren't whitespace.
    """
    if not text:
        return text

    result = []
    for char in text:
        code = ord(char)
        # Allow valid XML characters
        if code == 0x09 or code == 0x0A or code == 0x0D:  # Tab, LF, CR
            result.append(char)
        elif 0x20 <= code <= 0xD7FF:
            result.append(char)
        elif 0xE000 <= code <= 0xFFFD:
            result.append(char)
        # Skip NULL bytes and other control characters
        elif code == 0x00:
            continue  # Skip NULL bytes
        elif code < 0x20:
            continue  # Skip other control characters
        else:
            # For other characters, try to keep them (might be valid in some contexts)
            result.append(char)

    return "".join(result)


class FontStyle(BaseModel):
    font_family: str = "Times New Roman"
    size: int = 12
    bold: bool = False
    italic: bool = False
    underline: bool = False


class DocumentFormatting(BaseModel):
    """Simple bundle of font styles used throughout the document."""

    default_font: FontStyle = FontStyle()
    heading_font: FontStyle = FontStyle(size=16, bold=True, underline=True)
    subheading_font: FontStyle = FontStyle(size=14, bold=True)
    body_font: FontStyle = FontStyle()


def apply_font_style(run, font_style: FontStyle):
    """Apply font styling to a docx Run object."""
    run.font.name = font_style.font_family
    run.font.size = Pt(font_style.size)
    run.font.bold = font_style.bold
    run.font.italic = font_style.italic
    run.font.underline = font_style.underline
    # Ensure text color is always black unless specifically set elsewhere
    if not hasattr(run.font.color, "rgb") or run.font.color.rgb is None:
        run.font.color.rgb = RGBColor(0, 0, 0)


def add_page_borders(section):
    """
    Add borders to all pages in a section.
    Creates a clean professional border around page content.
    """
    try:
        # Create page border element
        sectPr = section._sectPr
        pgBorders = sectPr.find(qn("w:pgBorders"))

        if pgBorders is None:
            pgBorders = OxmlElement("w:pgBorders")
            # offsetFrom="text" makes borders relative to text, not page edge
            pgBorders.set(qn("w:offsetFrom"), "text")
            sectPr.append(pgBorders)

        # Define border sides
        for border_name in ("top", "left", "bottom", "right"):
            border = OxmlElement(f"w:{border_name}")
            border.set(qn("w:val"), "single")  # Single line border
            border.set(qn("w:sz"), "12")  # Border size (1/8 pt units, 12 = 1.5pt)
            border.set(qn("w:space"), "24")  # Space between border and text (pt)
            border.set(qn("w:color"), "000000")  # Black border
            pgBorders.append(border)

    except Exception as e:
        logger.error(f"Error adding page borders: {e}")


def add_page_numbers(section):
    """
    Add page numbers to footer of all pages in a section.
    Format: centered at bottom of page.
    """
    try:
        # Create footer if it doesn't exist
        footer = section.footer

        # Clear existing footer content
        for paragraph in footer.paragraphs:
            paragraph.clear()

        # Create paragraph for page number
        paragraph = (
            footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        )
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add page number field
        run = paragraph.add_run()

        # Create field for page number
        fldChar1 = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")

        instrText = OxmlElement("w:instrText")
        instrText.set(qn("xml:space"), "preserve")
        instrText.text = "PAGE"

        fldChar2 = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "separate")

        fldChar3 = OxmlElement("w:fldChar")
        fldChar3.set(qn("w:fldCharType"), "end")

        # Add elements to run
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        run._r.append(fldChar3)

        # Format page number
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"

    except Exception as e:
        logger.error(f"Error adding page numbers: {e}")


def add_apa_title_page(
    doc: Document,
    title: str,
    author: str = "Student Name",
    institution: str = "Institution Name",
    course: str = "Course Name",
    instructor: str = "Instructor Name",
    date: Optional[str] = None,
):
    """
    Add an APA 7th edition formatted title page.

    Args:
        doc: Document object
        title: Paper title
        author: Author name
        institution: Institution name
        course: Course name/number
        instructor: Instructor name
        date: Due date (defaults to current date)
    """
    from datetime import datetime

    if not date:
        date = datetime.now().strftime("%B %d, %Y")

    # Title (centered, bold, title case)
    # Position approximately 3-4 inches from top
    for _ in range(5):  # Add spacing from top
        doc.add_paragraph()

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.font.name = "Times New Roman"
    title_run.font.size = Pt(12)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing
    for _ in range(3):
        doc.add_paragraph()

    # Author name (centered)
    author_para = doc.add_paragraph()
    author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_run = author_para.add_run(author)
    author_run.font.name = "Times New Roman"
    author_run.font.size = Pt(12)
    author_run.font.color.rgb = RGBColor(0, 0, 0)

    # Institution (centered)
    inst_para = doc.add_paragraph()
    inst_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inst_run = inst_para.add_run(institution)
    inst_run.font.name = "Times New Roman"
    inst_run.font.size = Pt(12)
    inst_run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing
    for _ in range(2):
        doc.add_paragraph()

    # Course information (centered)
    course_para = doc.add_paragraph()
    course_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    course_run = course_para.add_run(course)
    course_run.font.name = "Times New Roman"
    course_run.font.size = Pt(12)
    course_run.font.color.rgb = RGBColor(0, 0, 0)

    # Instructor (centered)
    instructor_para = doc.add_paragraph()
    instructor_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    instructor_run = instructor_para.add_run(instructor)
    instructor_run.font.name = "Times New Roman"
    instructor_run.font.size = Pt(12)
    instructor_run.font.color.rgb = RGBColor(0, 0, 0)

    # Date (centered)
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_run = date_para.add_run(date)
    date_run.font.name = "Times New Roman"
    date_run.font.size = Pt(12)
    date_run.font.color.rgb = RGBColor(0, 0, 0)

    # Add page break after title page
    doc.add_page_break()


def add_table_of_contents(doc: Document):
    """Add a proper Word TOC element."""
    # Add TOC heading
    toc_heading = doc.add_heading("Table of Contents", level=0)
    toc_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in toc_heading.runs:
        run.font.name = "Arial"
        run.font.size = Pt(14)
        run.bold = True
        run.underline = True
        run.font.color.rgb = RGBColor(0, 0, 0)

    # Create TOC paragraph with proper field structure
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()

    # Create field begin
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")

    # Create instruction text
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'

    # Create field separate
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")

    # Create field end
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")

    # Add elements to run
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)

    # Add placeholder text
    placeholder_run = paragraph.add_run("Right-click to update field.")
    placeholder_run.font.italic = True
    placeholder_run.font.color.rgb = RGBColor(128, 128, 128)

    # Add field end to the placeholder run
    placeholder_run._r.append(fldChar3)

    # Set document to update fields on open
    settings = doc.settings
    settings_element = settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings_element.append(update_fields)

    # Add page break after TOC (respect global toggle)
    if os.getenv("ENABLE_PAGE_BREAKS", "false").lower() == "true":
        doc.add_page_break()


def generate_document(
    processed_sections,
    formatting: DocumentFormatting,
    title: str = "",
    enable_title_page: bool = True,
    enable_borders: bool = True,
    enable_page_numbers: bool = True,
    title_page_info: Optional[Dict[str, str]] = None,
    format_style: Optional[str] = None,
) -> Document:
    """
    Generate a DOCX document with proper formatting, optional title page, borders, and page numbers.

    Args:
        processed_sections: List of processed document sections
        formatting: Document formatting settings
        title: Document title (for title page)
        enable_title_page: Whether to include APA-style title page
        enable_borders: Whether to add page borders
        enable_page_numbers: Whether to add page numbers
        title_page_info: Optional dict with title page details (author, institution, etc.)
        apa_formatter: Optional APA formatter instance for managing citations and references
    """
    doc = Document()

    # Configure document styles
    styles = doc.styles

    # Configure default font
    style = styles["Normal"]
    font = style.font
    font.name = formatting.default_font.font_family
    font.size = Pt(formatting.default_font.size)
    font.color.rgb = RGBColor(0, 0, 0)  # Ensure body text is black

    # Configure heading styles
    try:
        heading1_style = styles["Heading 1"]
        heading1_font = heading1_style.font
        heading1_font.name = formatting.heading_font.font_family
        heading1_font.size = Pt(formatting.heading_font.size)
        heading1_font.bold = formatting.heading_font.bold
        heading1_font.underline = formatting.heading_font.underline
        heading1_font.color.rgb = RGBColor(0, 0, 0)
    except KeyError:
        pass  # Style might not exist

    try:
        heading2_style = styles["Heading 2"]
        heading2_font = heading2_style.font
        heading2_font.name = formatting.subheading_font.font_family
        heading2_font.size = Pt(formatting.subheading_font.size)
        heading2_font.bold = formatting.subheading_font.bold
        heading2_font.color.rgb = RGBColor(0, 0, 0)
    except KeyError:
        pass  # Style might not exist

    # Ensure Heading 3 is defined for all subsections (TOC-friendly)
    try:
        heading3_style = styles["Heading 3"]
        heading3_font = heading3_style.font
        # Use body font but bold, smaller than Heading 2
        heading3_font.name = formatting.body_font.font_family
        heading3_font.size = Pt(formatting.body_font.size)
        heading3_font.bold = True
        heading3_font.color.rgb = RGBColor(0, 0, 0)
    except KeyError:
        pass

    # Add APA-style title page if enabled
    if enable_title_page and (title or title_page_info):
        info = title_page_info or {}
        add_apa_title_page(
            doc,
            title=title or info.get("title", "Document Title"),
            author=info.get("author", "Author Name"),
            institution=info.get("institution", "Institution Name"),
            course=info.get("course", "Course Name"),
            instructor=info.get("instructor", "Instructor Name"),
            date=info.get("date", None),
        )

    # Apply page borders and numbers to all sections
    for section in doc.sections:
        if enable_borders:
            add_page_borders(section)
        if enable_page_numbers:
            add_page_numbers(section)

    # Add table of contents at the beginning
    add_table_of_contents(doc)

    # Group sections by main section to handle heading structure properly
    sections_by_main = {}
    for section in processed_sections:
        main_section = section.section_data.section
        if main_section not in sections_by_main:
            sections_by_main[main_section] = []
        sections_by_main[main_section].append(section)

    # Process sections with a simple, predictable heading hierarchy
    for main_section_key in sorted(sections_by_main.keys(), key=lambda x: str(x)):
        sections_group = sections_by_main[main_section_key]

        # Find the main section (without subsection)
        main_section_obj = None
        subsections = []

        for section in sections_group:
            if section.section_data.subsection is None:
                main_section_obj = section
            else:
                subsections.append(section)

        # Add main section heading only once
        if main_section_obj:
            section_data = main_section_obj.section_data
            main_heading_text = (
                section_data.title or section_data.section or ""
            ).strip()
            if main_heading_text:
                # Add page break before Level 1 heading (respect global toggle)
                if os.getenv("ENABLE_PAGE_BREAKS", "false").lower() == "true":
                    doc.add_page_break()
                heading = doc.add_heading(main_heading_text, level=1)
                for run in heading.runs:
                    apply_font_style(run, formatting.heading_font)
                    run.font.color.rgb = RGBColor(0, 0, 0)  # Black color

            # Add main section content if it exists and is not empty
            content = main_section_obj.content.strip()
            if (
                content and len(content) > 10
            ):  # Ensure it's not just whitespace or very short
                add_formatted_content(doc, content, formatting.body_font)
            else:
                logger.warning(
                    f"Section '{main_heading_text}' has no content or content is too short"
                )

            # Add graphs for main section if any
            if hasattr(main_section_obj, "graphs") and main_section_obj.graphs:
                for graph_info in main_section_obj.graphs:
                    graph_path = graph_info.get("path", "")
                    graph_desc = graph_info.get(
                        "description", f"Figure {graph_info.get('index', '')}"
                    )
                    if graph_path and os.path.exists(graph_path):
                        add_image_to_document(doc, graph_path, caption=graph_desc)
        # Add subsections ordered by their subsection label (if present)
        subsections = sorted(
            subsections,
            key=lambda s: str(getattr(s.section_data, "subsection", "") or ""),
        )
        for subsection in subsections:
            section_data = subsection.section_data

            sub_text = (section_data.subsection or "").strip()
            title_text = (section_data.title or "").strip()
            sub_heading_text = (
                f"{sub_text} {title_text}".strip() or title_text or sub_text
            )

            if sub_heading_text:
                heading = doc.add_heading(sub_heading_text, level=2)
                for run in heading.runs:
                    apply_font_style(run, formatting.subheading_font)
                    run.font.color.rgb = RGBColor(0, 0, 0)

            # Add subsection content if it exists and is not empty
            content = subsection.content.strip()
            if (
                content and len(content) > 10
            ):  # Ensure it's not just whitespace or very short
                add_formatted_content(doc, content, formatting.body_font)
            else:
                logger.warning(
                    f"Subsection '{sub_heading_text}' has no content or content is too short"
                )

            # Add graphs for this subsection if any
            if hasattr(subsection, "graphs") and subsection.graphs:
                for graph_info in subsection.graphs:
                    graph_path = graph_info.get("path", "")
                    graph_desc = graph_info.get(
                        "description", f"Figure {graph_info.get('index', '')}"
                    )
                    if graph_path and os.path.exists(graph_path):
                        add_image_to_document(doc, graph_path, caption=graph_desc)

        # No page breaks between sections - only after TOC and before Level 1 headings

    # Optionally include references section if provided
    if os.getenv("ENABLE_REFERENCES", "true").lower() == "true":
        citation_style = format_style or (
            title_page_info.get("format_style") if title_page_info else None
        )
        # Default to APA if no style specified
        if not citation_style or not citation_style.strip():
            citation_style = "APA"
            logger.info(
                "No citation style specified in generate_formatted_document, defaulting to APA"
            )
        add_references_section(
            doc, processed_sections, formatting, format_style=citation_style
        )

    return doc


def add_references_section(
    doc: Document,
    processed_sections,  # type: ignore[override] – simple runtime container list
    formatting: DocumentFormatting,
    format_style: Optional[str] = None,
) -> None:
    """Add a properly formatted references section using citation formatters."""
    # Collect all unique references from all sections
    all_references: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for section in processed_sections:
        for ref in getattr(section, "references", []) or []:
            url = ref.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_references.append(ref)

    if not all_references:
        return  # No references to add

    # Get citation formatter - default to APA if style is not specified
    # This ensures we always have a valid formatter instead of showing errors
    citation_style = (
        format_style.strip() if format_style and format_style.strip() else "APA"
    )
    logger.info(f"Using citation style: {citation_style} (provided: {format_style})")
    formatter = get_formatter(citation_style)

    # Add page break before references (Level 1 heading)
    if os.getenv("ENABLE_PAGE_BREAKS", "false").lower() == "true":
        doc.add_page_break()

    # Add References heading
    heading = doc.add_heading("References", level=1)
    for run in heading.runs:
        apply_font_style(run, formatting.heading_font)
        run.font.color.rgb = RGBColor(0, 0, 0)  # Black color

    # Format and add each reference
    for i, ref in enumerate(all_references, 1):
        ref_para = doc.add_paragraph()
        ref_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # Add index for numbered styles (IEEE)
        if format_style and format_style.lower() == "ieee":
            formatted_ref = formatter.format_reference(ref, index=i)
        else:
            formatted_ref = formatter.format_reference(ref)

        ref_run = ref_para.add_run(formatted_ref)
        apply_font_style(ref_run, formatting.body_font)
        ref_run.font.color.rgb = RGBColor(0, 0, 0)


def add_formatted_content(doc: Document, content: str, font_style: FontStyle) -> None:
    """Add body content to the document with equation support (inline and block) and markdown tables."""
    if not content:
        return

    # Sanitize content to remove NULL bytes and invalid XML characters
    content = sanitize_xml_text(content)

    # Process content - tables are handled within paragraph blocks
    _add_text_content(doc, content, font_style)

    # Add a blank paragraph for breathing room after each section
    doc.add_paragraph()


def _add_text_content(doc: Document, content: str, font_style: FontStyle) -> None:
    """Add text content with equation support (helper function)."""
    # First, check for images in the entire content (before splitting)
    # This ensures images are processed before other content
    images = parse_markdown_images(content)

    if images:
        # Found images - process them one by one
        current_pos = 0
        for image_info in images:
            # Process text before this image
            text_before = content[current_pos : image_info["start_pos"]].strip()
            if text_before:
                _add_text_content(doc, text_before, font_style)

            # Add the image
            image_path = image_info["path"]
            caption = image_info.get("caption", image_info.get("alt", ""))

            # Handle relative paths - try to resolve them
            # First check if it's already an absolute path
            if not os.path.isabs(image_path):
                # Try relative to current working directory
                if not os.path.exists(image_path):
                    # Try relative to document generation directory
                    # This assumes images might be in a standard location
                    possible_paths = [
                        image_path,
                        os.path.join(".", image_path),
                        os.path.join("..", image_path),
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break

            # Only add image if file exists
            if os.path.exists(image_path):
                add_image_to_document(doc, image_path, caption=caption)
            else:
                logger.warning(
                    f"Image file not found: {image_path}. Skipping image insertion."
                )

            current_pos = image_info["end_pos"]

        # Process remaining text after last image
        text_after = content[current_pos:].strip()
        if text_after:
            _add_text_content(doc, text_after, font_style)

        return

    # No images found - check for tables in the entire content (before splitting)
    # This ensures tables aren't broken by paragraph splitting
    table = parse_markdown_table(content)

    if table:
        # Found a table - split around it
        text_before = content[: table["start_pos"]].strip()
        text_after = content[table["end_pos"] :].strip()

        # Process text before table
        if text_before:
            _add_text_content(doc, text_before, font_style)

        # Add the table
        _add_markdown_table(doc, table["rows"], font_style)

        # Process text after table (recursively in case there are more tables)
        if text_after:
            _add_text_content(doc, text_after, font_style)

        return

    # No table found - split content by double newlines for paragraphs
    para_blocks = content.split("\n\n")

    for para_text in para_blocks:
        para_text = para_text.strip()
        if not para_text:
            continue

        # Check for images in this paragraph block (in case image wasn't detected earlier)
        images = parse_markdown_images(para_text)
        if images:
            # Found images in this paragraph - process them
            current_pos = 0
            for image_info in images:
                # Process text before this image
                text_before = para_text[current_pos : image_info["start_pos"]].strip()
                if text_before:
                    _add_single_paragraph(doc, text_before, font_style)

                # Add the image
                image_path = image_info["path"]
                caption = image_info.get("caption", image_info.get("alt", ""))

                # Handle relative paths
                if not os.path.isabs(image_path):
                    if not os.path.exists(image_path):
                        possible_paths = [
                            image_path,
                            os.path.join(".", image_path),
                            os.path.join("..", image_path),
                        ]
                        for path in possible_paths:
                            if os.path.exists(path):
                                image_path = path
                                break

                # Only add image if file exists
                if os.path.exists(image_path):
                    add_image_to_document(doc, image_path, caption=caption)
                else:
                    logger.warning(
                        f"Image file not found: {image_path}. Skipping image insertion."
                    )

                current_pos = image_info["end_pos"]

            # Process remaining text after last image
            text_after = para_text[current_pos:].strip()
            if text_after:
                _add_single_paragraph(doc, text_after, font_style)

            continue

        # Check for tables in this paragraph block (in case table wasn't detected earlier)
        table = parse_markdown_table(para_text)
        if table:
            # Found a table in this paragraph
            text_before = para_text[: table["start_pos"]].strip()
            text_after = para_text[table["end_pos"] :].strip()

            # Add text before table
            if text_before:
                _add_single_paragraph(doc, text_before, font_style)

            # Add the table
            _add_markdown_table(doc, table["rows"], font_style)

            # Add text after table
            if text_after:
                _add_single_paragraph(doc, text_after, font_style)

            continue

        # No images or tables - process as normal paragraph with equations
        _add_single_paragraph(doc, para_text, font_style)


def _add_single_paragraph(doc: Document, para_text: str, font_style: FontStyle) -> None:
    """Add a single paragraph with equation support."""
    # Parse equations in this paragraph
    equations = parse_equations(para_text)

    if not equations:
        # No equations, add as normal paragraph
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = paragraph.add_run(para_text)
        apply_font_style(run, font_style)
        run.font.color.rgb = RGBColor(0, 0, 0)
        return

    # Process paragraph with equations
    # Separate block and inline equations
    block_equations = [eq for eq in equations if eq["type"] == "block"]
    inline_equations = [eq for eq in equations if eq["type"] == "inline"]

    # If there are block equations, process them separately
    if block_equations:
        current_pos = 0
        for block_eq in block_equations:
            # Add text before block equation
            text_before = para_text[current_pos : block_eq["start_pos"]].strip()
            if text_before:
                paragraph = doc.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                _add_text_with_inline_equations(
                    doc, paragraph, text_before, inline_equations, font_style
                )
                doc.add_paragraph()  # Spacing before block equation

            # Add block equation (centered with spacing after)
            render_block_equation(doc, block_eq["latex"])
            current_pos = block_eq["end_pos"]

        # Add remaining text after last block equation (if any)
        text_after = para_text[current_pos:].strip()
        if text_after:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _add_text_with_inline_equations(
                doc, paragraph, text_after, inline_equations, font_style
            )
    else:
        # No block equations, process entire paragraph with inline equations
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _add_text_with_inline_equations(
            doc, paragraph, para_text, inline_equations, font_style
        )


def _add_markdown_table(
    doc: Document, rows: List[List[str]], font_style: FontStyle
) -> None:
    """
    Add a Word table from parsed markdown table rows.

    Args:
        doc: Document object
        rows: List of lists of cell contents (first row is header)
        font_style: Font style for table text
    """
    if not rows:
        return

    # Determine number of columns from first row
    num_cols = len(rows[0]) if rows else 0
    if num_cols == 0:
        return

    # Create Word table
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = "Light Grid Accent 1"  # Use a built-in table style

    # Fill table cells
    for row_idx, row_data in enumerate(rows):
        for col_idx in range(num_cols):
            cell = table.rows[row_idx].cells[col_idx]
            cell_text = row_data[col_idx] if col_idx < len(row_data) else ""

            # Sanitize cell text
            cell_text = sanitize_xml_text(cell_text)

            # Check if cell contains LaTeX notation (like |0⟩, |1⟩, or $...$)
            # For now, add as plain text (we can enhance this later to render LaTeX)
            cell_para = cell.paragraphs[0]
            cell_para.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            )

            # Try to render any inline equations in the cell
            from tools.common.math import parse_equations, sanitize_latex

            cell_equations = parse_equations(cell_text)

            if cell_equations:
                # Has equations - render with equation support
                _add_text_with_inline_equations(
                    doc, cell_para, cell_text, cell_equations, font_style
                )
            else:
                # No equations - just add text
                run = cell_para.add_run(cell_text)
                apply_font_style(run, font_style)
                run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing after table
    doc.add_paragraph()


def _add_text_with_inline_equations(
    doc: Document,
    paragraph: Any,
    text: str,
    inline_equations: List[Dict[str, Any]],
    font_style: FontStyle,
) -> None:
    """
    Add text to paragraph, replacing inline equations with rendered math.

    Args:
        doc: Document object
        paragraph: Paragraph object to add to
        text: Text content
        inline_equations: List of inline equation dicts
        font_style: Font style for text
    """
    # Sanitize text to remove NULL bytes and invalid XML characters
    text = sanitize_xml_text(text)

    # Filter inline equations that are in this text
    # We need to match equations by their position in the original text
    # For simplicity, parse equations again from this text segment
    segment_equations = [eq for eq in parse_equations(text) if eq["type"] == "inline"]

    if not segment_equations:
        # No inline equations, just add text
        run = paragraph.add_run(text)
        apply_font_style(run, font_style)
        run.font.color.rgb = RGBColor(0, 0, 0)
        return

    # Process text with inline equations
    current_pos = 0
    for eq in segment_equations:
        # Add text before equation
        text_before = text[current_pos : eq["start_pos"]]
        if text_before:
            run = paragraph.add_run(text_before)
            apply_font_style(run, font_style)
            run.font.color.rgb = RGBColor(0, 0, 0)

        # Add inline equation
        try:
            # Add space before if there's preceding text
            if text_before and text_before.rstrip():
                run = paragraph.add_run(" ")
                apply_font_style(run, font_style)

            # Sanitize LaTeX before rendering
            from tools.common.math import sanitize_latex

            sanitized_latex = sanitize_latex(eq["latex"])

            import math2docx

            math2docx.add_math(paragraph, sanitized_latex)

            # Add space after equation
            run = paragraph.add_run(" ")
            apply_font_style(run, font_style)
        except Exception as e:
            logger.warning(
                f"Failed to render inline equation '{eq['latex']}': {e}. Using plain text fallback."
            )
            # Sanitize fallback text
            from tools.common.math import sanitize_latex

            fallback_text = sanitize_latex(f" [${eq['latex']}$] ")
            run = paragraph.add_run(fallback_text)
            apply_font_style(run, font_style)
            run.font.color.rgb = RGBColor(0, 0, 0)

        current_pos = eq["end_pos"]

    # Add remaining text after last equation
    text_after = text[current_pos:]
    if text_after:
        run = paragraph.add_run(text_after)
        apply_font_style(run, font_style)
        run.font.color.rgb = RGBColor(0, 0, 0)


class _SectionData:
    """Lightweight container for section metadata used by `generate_document`."""

    def __init__(self, section: str, subsection: Optional[str] = None, title: str = ""):
        self.section = section
        self.subsection = subsection
        self.title = title


class _ProcessedSection:
    """Internal representation matching what `generate_document` expects."""

    def __init__(
        self,
        section: str,
        subsection: Optional[str],
        title: str,
        content: str,
        references: Optional[List[Dict[str, Any]]] = None,
        graphs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.section_data = _SectionData(
            section=section, subsection=subsection, title=title
        )
        self.content = content or ""
        self.references: List[Dict[str, Any]] = references or []
        self.graphs: List[Dict[str, Any]] = graphs or []


class GeneratedDocument(BaseModel):
    """Metadata about a generated DOCX document."""

    file_path: str = Field(
        ...,
        description="Absolute or relative filesystem path to the generated .docx file.",
    )
    title: str = Field(
        "",
        description="Title supplied for the generated document.",
    )
    num_sections: int = Field(
        0,
        description="Number of sections that were rendered into the document.",
    )


@tool
def generate_formatted_document(
    title: str,
    sections: List[Dict[str, Any]],
    output_filename: str = "generated_document.docx",
    enable_title_page: bool = True,
    enable_borders: bool = True,
    enable_page_numbers: bool = True,
    format_style: Optional[str] = None,
) -> GeneratedDocument:
    """
    Generate a fully formatted Word document (.docx) from structured section data.

    Args:
        title: Overall title of the document.
        sections: List of dictionaries, each with at least:
            - ``section``: Top‑level section label (e.g., ``"1"`` or ``"Introduction"``)
            - ``subsection``: Optional subsection label (e.g., ``"1.1"``) (optional)
            - ``title``: Human‑readable title for the section/subsection (optional)
            - ``content``: Body text for the section (plain text or lightly marked up)
            - ``references``: Optional list of reference dicts with keys like
              ``"url"``, ``"title"``, ``"snippet"``, etc.
        output_filename: Desired filename for the generated document. If this is a
            relative path, it will be created inside the ``generated_docs`` directory.
            ``.docx`` will be appended automatically if missing.
        enable_title_page: Whether to include an APA‑style title page.
        enable_borders: Whether to draw borders around each page.
        enable_page_numbers: Whether to add centered page numbers in the footer.
        format_style: Citation style for references section (IEEE, APA, MLA, Chicago).
            If None, uses a generic format.

    Returns:
        GeneratedDocument: Metadata about the saved document, including its path.
    """
    processed_sections: List[_ProcessedSection] = []

    for raw in sections:
        section_label = str(raw.get("section", "")).strip()
        subsection_label = raw.get("subsection")
        section_title = str(raw.get("title", "")).strip()
        content = str(raw.get("content", "") or "")
        references = raw.get("references") or []

        graphs = raw.get("graphs") or []
        processed_sections.append(
            _ProcessedSection(
                section=section_label,
                subsection=subsection_label,
                title=section_title,
                content=content,
                references=references,
                graphs=graphs,
            )
        )

    formatting = DocumentFormatting()

    # Basic title page info – can be extended by callers later if needed
    title_page_info = {"title": title} if title else None

    doc = generate_document(
        processed_sections=processed_sections,
        formatting=formatting,
        title=title,
        enable_title_page=enable_title_page,
        enable_borders=enable_borders,
        enable_page_numbers=enable_page_numbers,
        title_page_info=title_page_info,
        format_style=format_style,
    )

    # Work out where to save the file
    base_dir = os.getenv("DOC_OUTPUT_DIR", "generated_docs")

    if not output_filename.lower().endswith(".docx"):
        output_filename = f"{output_filename}.docx"

    if os.path.isabs(output_filename):
        output_path = output_filename
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    else:
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, output_filename)

    doc.save(output_path)

    return GeneratedDocument(
        file_path=output_path,
        title=title,
        num_sections=len(processed_sections),
    )
