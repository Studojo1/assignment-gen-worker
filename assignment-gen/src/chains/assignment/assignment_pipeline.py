"""
Project: Studojo Assignment Generator
File: src/chains/assignment/assignment_pipeline.py
Author: Rithul Kamesh
Created: 17 November 2025
Description: High-level pipeline for generating an assignment once the spec is READY.

Steps:
- For now, generate content for a single synthetic section using the content chain.
- Optionally humanize the content using Rephrasy.
- Generate a DOCX via the existing document builder.
- Run embeddings-based uniqueness check on the final text.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .assignment_content import SubsectionSpec, build_assignment_content_chain
from .search_utils import (
    build_legacy_search_query,
    extract_search_topic,
    generate_search_query_with_llm,
)
from models.assignment import AssignmentSpec
from tools.common.doc import (
    generate_formatted_document as generate_formatted_document_tool,
)
from tools.common.graph import (
    generate_matplotlib_graph as generate_matplotlib_graph_tool,
    generate_content_aware_graph as generate_content_aware_graph_tool,
    analyze_content_for_graphs,
)
from tools.common.rephrasy import RephrasyService
from tools.common.uniqueness import UniquenessChecker, UniquenessResult
from tools.common.web import search_and_fetch

logger = logging.getLogger(__name__)


@dataclass
class AssignmentGenerationResult:
    doc_path: str
    full_text: str
    uniqueness: Optional[UniquenessResult]


def build_assignment_pipeline(
    llm: BaseChatModel,
    rephrasy: Optional[RephrasyService] = None,
    uniqueness_checker: Optional[UniquenessChecker] = None,
):
    """
    Build a simple callable that runs section content -> humanizer -> DOCX + uniqueness.

    For now, this is intentionally minimal and oriented around a single section:
    - section_number: "1"
    - section_title: derived from assignment_type or a generic label
    - subsections: just one synthetic subsection key "body"
    """

    content_chain = build_assignment_content_chain(llm)
    rephrasy_service = rephrasy or RephrasyService()
    checker = uniqueness_checker or UniquenessChecker()

    async def _generate_assignment_title(
        spec: AssignmentSpec,
        content_preview: str,
    ) -> str:
        """
        Generate a smart, academic title for the assignment based on spec and content.

        Args:
            spec: Assignment specification
            content_preview: Preview of generated content (first section or summary)

        Returns:
            Generated title string
        """
        try:
            title_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an academic writing assistant. Generate a concise, professional title for an academic assignment. The title should be specific to the content, not generic.",
                    ),
                    (
                        "human",
                        """Generate a title for this assignment based on the following information:

Assignment Type: {assignment_type}
Course: {course_name}
Description: {description}

Content Preview (first ~500 words):
{content_preview}

Generate a single, concise title (10-15 words maximum) that accurately reflects the assignment topic and content. Do not include quotation marks or any formatting. Return only the title text.""",
                    ),
                ]
            )

            title_chain = title_prompt | llm
            response = await title_chain.ainvoke(
                {
                    "assignment_type": spec.assignment_type or "Assignment",
                    "course_name": spec.course_name or "",
                    "description": spec.description or "",
                    "content_preview": content_preview[:500] if content_preview else "",
                }
            )

            title = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            # Remove any quotes or formatting
            title = title.strip("\"'")

            if title and len(title) > 5:
                logger.info(f"Generated assignment title: {title}")
                return title
            else:
                logger.warning("Generated title was too short, using fallback")
                return spec.assignment_type or "Assignment"
        except Exception as e:
            logger.warning(f"Title generation failed: {e}, using fallback")
            return spec.assignment_type or "Assignment"

    async def run(
        spec: AssignmentSpec,
        *,
        outline: Optional[Dict[str, Any]] = None,
        courseplan_analysis: str = "",
        user_responses: str = "",
        corpus_texts: Optional[List[str]] = None,
        output_filename: str = "assignment.docx",
    ) -> AssignmentGenerationResult:
        # 1. Use provided outline or generate a synthetic single section.
        if outline and "sections" in outline:
            sections_from_outline = outline["sections"]
        else:
            # Fallback: build a synthetic single section if no outline provided.
            target_words_total = spec.length_words or int(
                (spec.length_pages or 5.0) * 300
            )
            target_pages = float(spec.length_pages or target_words_total / 300.0)
            sections_from_outline = [
                {
                    "id": "1",
                    "title": spec.assignment_type or "Assignment",
                    "target_words": target_words_total,
                    "subsections": [
                        {
                            "id": "1.1",
                            "title": "Main Discussion",
                            "target_words": target_words_total,
                        }
                    ],
                }
            ]

        # Rough target words for context.
        target_words_total = spec.length_words or int((spec.length_pages or 5.0) * 300)

        # Build citation instruction based on format_style
        citation_instruction = ""
        if spec.format_style:
            style_lower = spec.format_style.lower()
            if style_lower == "ieee":
                citation_instruction = (
                    "Cite all sources using IEEE format. Use in-text citations like [1] "
                    "and include full references in the References section."
                )
            elif style_lower == "apa":
                citation_instruction = (
                    "Cite all sources using APA format. Use in-text citations like (Author, Year) "
                    "and include full references in the References section."
                )
            elif style_lower == "mla":
                citation_instruction = (
                    "Cite all sources using MLA format. Use in-text citations like (Author) "
                    "and include full references in the References section."
                )
            elif style_lower == "chicago":
                citation_instruction = (
                    "Cite all sources using Chicago format. Use in-text citations like (Author Year) "
                    "and include full references in the References section."
                )
            else:
                citation_instruction = (
                    f"Cite all sources using {spec.format_style} format. "
                    "Include full references in the References section."
                )
        else:
            citation_instruction = "Cite all sources appropriately and include full references in the References section."

        # 1.5. Use web search results from spec (already collected during input phase)
        all_web_references: List[Dict[str, Any]] = []
        web_search_block = ""

        if spec.allow_web_search and spec.web_search_results:
            # Use the web search results that were collected during input collection
            from urllib.parse import urlparse

            web_parts = []
            for i, result in enumerate(spec.web_search_results, 1):
                title = result.get("title", "Unknown Title")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                content = result.get("content", "")

                web_parts.append(
                    f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n"
                    f"Content: {content[:500]}..."
                    if len(content) > 500
                    else f"Content: {content}"
                )

                # Extract website name from URL for author field
                try:
                    parsed_url = urlparse(url)
                    netloc = parsed_url.netloc.replace("www.", "")
                    parts = netloc.split(".")

                    # Smarter author extraction: ignore common subdomains
                    # e.g. support.google.com -> google, docs.python.org -> python
                    ignore_subdomains = {
                        "support",
                        "help",
                        "docs",
                        "developer",
                        "api",
                        "learn",
                        "en",
                        "www",
                    }

                    if len(parts) >= 2:
                        # If first part is in ignore list, take the second part
                        if parts[0].lower() in ignore_subdomains and len(parts) > 2:
                            website_name = parts[1]
                        else:
                            website_name = parts[0]
                    else:
                        website_name = parts[0] if parts else "Unknown"

                    # Capitalize first letter and format nicely
                    website_name = (
                        website_name.capitalize() if website_name else "Unknown"
                    )
                    website_domain = netloc if netloc else ""
                except Exception as e:
                    logger.warning(f"Error parsing URL {url}: {e}")
                    website_name = "Unknown"
                    website_domain = ""

                # Try to extract year from snippet or content
                import re

                year = ""
                # Look for 4-digit year in snippet (2000-2029)
                year_match = re.search(r"\b(20[0-2][0-9])\b", snippet)
                if year_match:
                    year = year_match.group(1)

                # Clean title (remove " | Website", " - Website", etc.)
                clean_title = title
                if website_name and website_name.lower() in title.lower():
                    # Remove suffix like " | Upwork" or " - Upwork"
                    clean_title = re.sub(
                        f"\\s*[|:-]\\s*{re.escape(website_name)}.*$",
                        "",
                        clean_title,
                        flags=re.IGNORECASE,
                    )

                # Check for duplicates before adding
                is_duplicate = False
                for existing in all_web_references:
                    # Check URL match
                    if existing["url"] == url:
                        is_duplicate = True
                        break
                    # Check Title match (fuzzy)
                    if clean_title.lower() == existing["title"].lower():
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Store reference for document builder
                    all_web_references.append(
                        {
                            "title": clean_title.strip(),
                            "url": url,
                            "snippet": snippet,
                            "content": content,
                            "index": len(all_web_references)
                            + 1,  # Re-index based on actual list
                            "_index": len(all_web_references) + 1,
                            "website": website_domain,  # Full domain
                            "author": website_name,  # Use website name as author fallback
                            "year": year,
                            "date": year,  # Fallback for formatters expecting date
                        }
                    )

            web_search_block = "\n\n".join(web_parts)
            logger.info(
                f"Using {len(spec.web_search_results)} web sources from input collection phase"
            )
        elif spec.allow_web_search and not spec.web_search_results:
            logger.warning(
                "Web search is enabled but no results were found during input collection. "
                "Attempting web search retry during generation..."
            )
            # Retry web search if it failed during input collection
            # This can happen if description wasn't fully populated or search failed transiently
            if spec.description or (
                spec.uploaded_files and len(spec.uploaded_files) > 0
            ):
                try:
                    web_results = []

                    # Build uploaded files context if available
                    uploaded_files_context = ""
                    if spec.uploaded_files and len(spec.uploaded_files) > 0:
                        file_parts = []
                        for i, file_info in enumerate(spec.uploaded_files, 1):
                            filename = file_info.get("filename", f"File {i}")
                            extracted_text = file_info.get("extracted_text", "")
                            if extracted_text:
                                # Show preview of file content (first 1000 chars for better context)
                                preview = extracted_text[:1000]
                                file_parts.append(
                                    f"=== Uploaded File {i}: {filename} ===\n"
                                    f"Content preview:\n{preview}"
                                    f"{'...' if len(extracted_text) > 1000 else ''}\n"
                                    f"Total length: {len(extracted_text)} characters\n"
                                )
                        if file_parts:
                            uploaded_files_context = "\n".join(file_parts)

                    # Use LLM for search query generation
                    search_query = await generate_search_query_with_llm(
                        llm,
                        spec.description or "",
                        spec.course_name,
                        uploaded_files_context if uploaded_files_context else None,
                    )
                    if search_query:
                        logger.info(f"LLM generated retry search query: {search_query}")

                    if not search_query:
                        search_query = build_legacy_search_query(
                            spec.description, spec.course_name
                        )
                        if search_query:
                            logger.info(
                                "LLM returned empty query during retry; falling back to legacy heuristics."
                            )

                    if not search_query:
                        logger.warning(
                            "Unable to derive a concise search query from description; skipping retry."
                        )
                    else:
                        logger.info(f"Retrying web search with query: {search_query}")
                        # Add timeout handling - 30 seconds should be enough for web search
                        try:
                            web_results = await asyncio.wait_for(
                                search_and_fetch(search_query, max_results=5),
                                timeout=30.0,
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                f"Web search retry timed out after 30 seconds for query: {search_query}"
                            )
                            logger.warning(
                                "Continuing without web sources due to timeout."
                            )
                            web_results = []
                        except Exception as timeout_error:
                            logger.error(
                                f"Web search retry failed with error: {timeout_error}",
                                exc_info=True,
                            )
                            logger.warning("Continuing without web sources.")
                            web_results = []

                    if web_results:
                        logger.info(
                            f"Retry web search found {len(web_results)} results"
                        )
                        # Store results in spec for future use
                        spec.web_search_results = web_results[:5]

                        # Process results same as above
                        from urllib.parse import urlparse

                        web_parts = []
                        for i, result in enumerate(web_results[:5], 1):
                            title = result.get("title", "Unknown Title")
                            url = result.get("url", "")
                            snippet = result.get("snippet", "")
                            content = result.get("content", "")

                            web_parts.append(
                                f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n"
                                f"Content: {content[:500]}..."
                                if len(content) > 500
                                else f"Content: {content}"
                            )

                            try:
                                parsed_url = urlparse(url)
                                netloc = parsed_url.netloc.replace("www.", "")
                                website_name = (
                                    netloc.split(".")[0] if netloc else "Unknown"
                                )
                                website_name = (
                                    website_name.capitalize()
                                    if website_name
                                    else "Unknown"
                                )
                                website_domain = netloc if netloc else ""
                            except Exception as e:
                                logger.warning(f"Error parsing URL {url}: {e}")
                                website_name = "Unknown"
                                website_domain = ""

                            all_web_references.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "content": content,
                                    "index": i,
                                    "_index": i,
                                    "website": website_domain,
                                    "author": website_name,
                                }
                            )

                        web_search_block = "\n\n".join(web_parts)
                        logger.info(
                            f"Successfully retried web search: found {len(web_results)} sources"
                        )
                    else:
                        logger.warning(
                            f"Web search retry also returned no results for query: {search_query}"
                        )
                except Exception as e:
                    logger.error(f"Web search retry failed: {e}", exc_info=True)
                    logger.warning("Continuing without web sources.")
            else:
                logger.warning("Cannot retry web search: no description available.")

        # 1.6. Graph generation moved to AFTER content generation (see step 2.5)
        generated_graphs: List[Dict[str, Any]] = []
        graph_context_block = ""

        # 2. Generate content for each section in the outline.
        all_sections_payload: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []

        for section_data in sections_from_outline:
            section_id = str(section_data.get("id", ""))
            section_title = str(section_data.get("title", ""))
            section_target_words = int(section_data.get("target_words", 500))
            subsections_list = section_data.get("subsections", [])

            # Convert outline subsections to SubsectionSpec format.
            subsections: List[SubsectionSpec] = []
            for sub in subsections_list:
                sub_target_words = int(sub.get("target_words", 200))
                sub_pages = max(0.1, sub_target_words / 300.0)  # Approximate pages.
                subsections.append(
                    {
                        "key": str(sub.get("id", "")).replace(
                            ".", "_"
                        ),  # Safe key for JSON.
                        "title": str(sub.get("title", "")),
                        "pages": sub_pages,
                    }
                )

            # If no subsections, add a synthetic "main" subsection.
            if not subsections:
                section_pages = max(0.1, section_target_words / 300.0)
                subsections = [
                    {"key": "main", "title": section_title, "pages": section_pages}
                ]

            # Log what subsection keys we're requesting
            expected_keys = ["main"] + [
                sub["key"] for sub in subsections if sub["key"] != "main"
            ]
            logger.info(
                f"Section {section_id}: Requesting content for keys: {expected_keys}"
            )
            logger.debug(f"Section {section_id}: Subsection specs: {subsections}")
            # Also print to stdout for worker visibility
            print(f"[pipeline] Processing Section {section_id}: '{section_title}' ({section_target_words} words)", flush=True)
            if subsections:
                print(f"[pipeline]   Subsections: {', '.join([s.get('title', s.get('key', '?')) for s in subsections])}", flush=True)

            # Generate content for this section.
            print(f"[pipeline] Generating content for Section {section_id}...", flush=True)
            try:
                json_content: Dict[str, str] = await content_chain.ainvoke(
                    {
                        "section_number": section_id,
                        "section_title": section_title,
                        "target_pages": max(0.1, section_target_words / 300.0),
                        "target_words_total": section_target_words,
                        "courseplan_analysis": courseplan_analysis
                        or (spec.description or ""),
                        "user_responses": user_responses,
                        "graph_context_block": graph_context_block,
                        "web_search_block": web_search_block,
                        "excel_context_snippet": "",
                        "citation_instruction": citation_instruction,
                        "dynamic_forbidden_block": "",
                        "subsections": subsections,
                    }
                )
            except Exception as content_error:
                logger.error(
                    f"Failed to generate content for section {section_id}: {content_error}",
                    exc_info=True,
                )
                # Create empty content dict to prevent crashes
                json_content = {"main": ""}
                logger.warning(
                    f"Section {section_id} will have empty content due to generation error"
                )

            # Log what we actually received
            received_keys = list(json_content.keys())
            missing_keys = set(expected_keys) - set(received_keys)
            if missing_keys:
                logger.warning(
                    f"Section {section_id}: Missing keys in JSON response: {missing_keys}"
                )
                logger.warning(
                    f"Section {section_id}: Expected {len(expected_keys)} keys, got {len(received_keys)} keys"
                )
                logger.warning(f"Section {section_id}: Expected keys: {expected_keys}")
                logger.warning(f"Section {section_id}: Received keys: {received_keys}")
            else:
                logger.info(
                    f"Section {section_id}: All expected keys present in JSON response"
                )
                print(f"[pipeline] ✓ Section {section_id} content generated successfully", flush=True)

            # Extract main section content (if present).
            main_text = json_content.get("main", "")
            if main_text and main_text.strip():
                logger.info(
                    f"Section {section_id} main content length: {len(main_text)}"
                )
                # Determine if this section should have graphs
                # Add graphs to sections that mention graphs/visualizations/samples/figures
                section_title_lower = section_title.lower()
                should_have_graphs = generated_graphs and any(
                    keyword in section_title_lower
                    for keyword in [
                        "graph",
                        "visualization",
                        "figure",
                        "sample",
                        "diagram",
                        "chart",
                        "plot",
                        "mind map",
                        "mindmap",
                        "screenshot",
                        "image",
                    ]
                )
                section_graphs = generated_graphs if should_have_graphs else []
                if section_graphs:
                    logger.info(
                        f"Attaching {len(section_graphs)} graphs to section '{section_title}' (keyword match)"
                    )
                    for g in section_graphs:
                        logger.debug(
                            f"  - Graph: {g.get('description', 'unknown')} at {g.get('path', 'unknown')}"
                        )
                all_sections_payload.append(
                    {
                        "section": section_id,
                        "subsection": None,
                        "title": section_title,
                        "content": main_text,
                        "references": (
                            all_web_references if spec.allow_web_search else []
                        ),
                        "graphs": section_graphs,
                    }
                )
                all_text_parts.append(main_text)
            else:
                logger.warning(
                    f"Section {section_id} has no main content (empty or missing)"
                )
                logger.warning(
                    f"Section {section_id} main_text value: {repr(main_text[:100]) if main_text else 'None'}"
                )

            # Extract subsection content and add to payload.
            for sub in subsections_list:
                sub_id = str(sub.get("id", ""))
                sub_title = str(sub.get("title", ""))
                sub_key = sub_id.replace(".", "_")

                # Try both the expected key and alternative formats
                sub_content = json_content.get(sub_key, "")
                if not sub_content or not sub_content.strip():
                    # Try alternative key formats
                    alt_key = sub_id  # Try with dot
                    sub_content = json_content.get(alt_key, "")
                    if not sub_content or not sub_content.strip():
                        # Try with underscore but different format
                        alt_key2 = sub_id.replace(".", "_").lower()
                        sub_content = json_content.get(alt_key2, "")

                if sub_content and sub_content.strip():
                    logger.info(
                        f"Section {section_id} subsection {sub_key} content length: {len(sub_content)}"
                    )
                    # Determine if this subsection should have graphs
                    # Add graphs to subsections that mention graphs/visualizations
                    sub_title_lower = sub_title.lower()
                    should_have_graphs = generated_graphs and any(
                        keyword in sub_title_lower
                        for keyword in [
                            "graph",
                            "visualization",
                            "figure",
                            "sample",
                            "diagram",
                            "chart",
                            "plot",
                            "plot",
                            "truth table",
                            "mind map",
                            "mindmap",
                            "screenshot",
                            "image",
                        ]
                    )
                    sub_graphs = generated_graphs if should_have_graphs else []
                    if sub_graphs:
                        logger.info(
                            f"Attaching {len(sub_graphs)} graphs to subsection '{sub_title}' (keyword match)"
                        )
                        for g in sub_graphs:
                            logger.debug(
                                f"  - Graph: {g.get('description', 'unknown')} at {g.get('path', 'unknown')}"
                            )
                    all_sections_payload.append(
                        {
                            "section": section_id,
                            "subsection": sub_id,
                            "title": sub_title,
                            "content": sub_content,
                            "references": (
                                all_web_references if spec.allow_web_search else []
                            ),
                            "graphs": sub_graphs,
                        }
                    )
                    all_text_parts.append(sub_content)
                else:
                    logger.warning(
                        f"  Sub content value: {repr(sub_content[:100]) if sub_content else 'None'}"
                    )

        # 2.5. Generate content-aware graphs AFTER content is generated
        # Only generate graphs if user explicitly confirmed they want graphs
        # (allows_graphs must be explicitly True, not inferred from keywords)
        should_generate_graphs = spec.allows_graphs is True

        if should_generate_graphs:
            logger.info(
                "Generating content-aware graphs based on actual assignment content..."
            )
            doc_output_dir = os.getenv("DOC_OUTPUT_DIR", "generated_docs")
            os.makedirs(doc_output_dir, exist_ok=True)

            # Track seen graphs to prevent duplicates (one graph per unique title+data combination)
            # Key: (title, frozenset of data_points items) -> graph_info dict
            seen_graphs: Dict[tuple, Dict[str, Any]] = {}

            # Analyze each section/subsection content and generate relevant graphs
            for entry in all_sections_payload:
                content = entry.get("content", "")
                section_title = entry.get("title", "")

                if not content or len(content.strip()) < 100:
                    continue  # Skip sections with insufficient content

                # Analyze content to find relevant graphs
                try:
                    # Pass web sources for validation if available
                    web_sources_for_validation = (
                        all_web_references if spec.allow_web_search else None
                    )
                    graph_specs = await analyze_content_for_graphs(
                        content,
                        max_graphs=1,
                        web_sources=web_sources_for_validation,
                        llm=llm,
                    )

                    if graph_specs:
                        spec_obj = graph_specs[0]
                        logger.info(
                            f"Analyzed section '{section_title}' and found relevant graph: {spec_obj.title}"
                        )

                        # Create deduplication key: (title, hash of data_points)
                        # Use frozenset to make data_points hashable for comparison
                        data_points_items = frozenset(
                            sorted(spec_obj.data_points.items())
                        )
                        graph_key = (spec_obj.title, data_points_items)

                        # Check if we've already generated this graph
                        if graph_key in seen_graphs:
                            # Skip duplicate graph - each graph should appear exactly once
                            logger.info(
                                f"Skipping duplicate graph '{spec_obj.title}' with same data for section '{section_title}' (already generated)"
                            )
                            continue
                        else:
                            # Generate new graph
                            graph_filename = f"graph_{section_title.replace(' ', '_')}_{int(time.time())}.png"
                            graph_output_path = os.path.join(
                                os.path.abspath(doc_output_dir), graph_filename
                            )

                            result = await generate_content_aware_graph_tool.coroutine(
                                content=content,
                                section_title=section_title,
                                output_filename=graph_output_path,
                                web_sources=web_sources_for_validation,
                            )

                            if (
                                result.success
                                and result.image_path
                                and os.path.exists(result.image_path)
                            ):
                                graph_path = os.path.abspath(result.image_path)
                                graph_info = {
                                    "path": graph_path,
                                    "description": spec_obj.title,
                                    "section": entry.get("section"),
                                    "subsection": entry.get("subsection"),
                                }

                                # Store in seen_graphs for deduplication
                                seen_graphs[graph_key] = graph_info

                                # Add to this entry's graphs
                                if "graphs" not in entry:
                                    entry["graphs"] = []
                                entry["graphs"].append(graph_info)
                                generated_graphs.append(graph_info)

                                logger.info(
                                    f"Generated content-aware graph for section '{section_title}': {graph_path}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to generate content-aware graph for section '{section_title}': {result.error}"
                                )
                    else:
                        logger.debug(
                            f"No relevant graphs found for section '{section_title}'"
                        )

                except Exception as e:
                    logger.error(
                        f"Error generating content-aware graph for section '{section_title}': {e}",
                        exc_info=True,
                    )

            # If we still have no graphs and user requested graphs, generate at least one based on overall content
            if not generated_graphs and all_text_parts:
                try:
                    logger.info(
                        "No section-specific graphs generated, creating graph from overall content..."
                    )
                    full_content_preview = "\n\n".join(
                        all_text_parts[:3]
                    )  # Use first 3 sections

                    web_sources_for_validation = (
                        all_web_references if spec.allow_web_search else None
                    )
                    graph_specs = await analyze_content_for_graphs(
                        full_content_preview,
                        max_graphs=1,
                        web_sources=web_sources_for_validation,
                        llm=llm,
                    )
                    if graph_specs:
                        spec_obj = graph_specs[0]

                        # Check for duplicates using same key format
                        data_points_items = frozenset(
                            sorted(spec_obj.data_points.items())
                        )
                        graph_key = (spec_obj.title, data_points_items)

                        if graph_key in seen_graphs:
                            # Skip duplicate graph - each graph should appear exactly once
                            logger.info(
                                f"Skipping duplicate graph '{spec_obj.title}' for overview (already generated)"
                            )
                        else:
                            # Generate new graph
                            graph_filename = f"graph_overview_{int(time.time())}.png"
                            graph_output_path = os.path.join(
                                os.path.abspath(doc_output_dir), graph_filename
                            )

                            result = await generate_content_aware_graph_tool.coroutine(
                                content=full_content_preview,
                                section_title="Overview",
                                output_filename=graph_output_path,
                                web_sources=web_sources_for_validation,
                            )

                            if (
                                result.success
                                and result.image_path
                                and os.path.exists(result.image_path)
                            ):
                                graph_path = os.path.abspath(result.image_path)
                                graph_info = {
                                    "path": graph_path,
                                    "description": spec_obj.title,
                                    "section": None,
                                    "subsection": None,
                                }

                                # Store in seen_graphs for deduplication
                                seen_graphs[graph_key] = graph_info
                                generated_graphs.append(graph_info)

                                # Add to first section with substantial content
                                for entry in all_sections_payload:
                                    content = entry.get("content", "")
                                    if content and len(content.strip()) > 100:
                                        if "graphs" not in entry:
                                            entry["graphs"] = []
                                        entry["graphs"].append(graph_info)
                                        break

                                logger.info(f"Generated overview graph: {graph_path}")
                except Exception as e:
                    logger.error(f"Error generating overview graph: {e}", exc_info=True)

        # Fallback: If graphs were generated but not attached to any section (due to strict keyword matching),
        # attach them to appropriate sections with improved logic
        if generated_graphs:
            graphs_attached = any(entry.get("graphs") for entry in all_sections_payload)
            if not graphs_attached and all_sections_payload:
                logger.warning(
                    f"Graphs were generated but not attached to any section. Using improved fallback attachment."
                )
                logger.info(
                    f"Generated {len(generated_graphs)} graphs: {[g.get('description', 'unknown') for g in generated_graphs]}"
                )

                # Strategy: Attach graphs to sections that make sense
                # 1. Try to find sections with data/analysis/results keywords
                preferred_keywords = [
                    "data",
                    "analysis",
                    "results",
                    "findings",
                    "discussion",
                    "methodology",
                    "method",
                    "approach",
                    "evaluation",
                    "assessment",
                ]

                target_entries = []
                for entry in all_sections_payload:
                    title_lower = entry.get("title", "").lower()
                    content = entry.get("content", "")
                    # Check if section title or content mentions preferred keywords
                    if any(keyword in title_lower for keyword in preferred_keywords):
                        if content and len(content.strip()) > 50:
                            target_entries.append(entry)

                # 2. If no preferred sections found, use sections with substantial content
                if not target_entries:
                    for entry in all_sections_payload:
                        title_lower = entry.get("title", "").lower()
                        # Skip introduction/abstract/conclusion
                        if not any(
                            k in title_lower
                            for k in [
                                "introduction",
                                "abstract",
                                "conclusion",
                                "reference",
                            ]
                        ):
                            content = entry.get("content", "")
                            if content and len(content.strip()) > 100:
                                target_entries.append(entry)

                # 3. If still nothing, use first section with any content
                if not target_entries:
                    for entry in all_sections_payload:
                        content = entry.get("content", "")
                        if content and len(content.strip()) > 10:
                            target_entries.append(entry)
                            break

                # 4. Last resort: use first section
                if not target_entries:
                    target_entries = (
                        [all_sections_payload[0]] if all_sections_payload else []
                    )

                # Distribute graphs across target sections
                if target_entries:
                    graphs_per_section = max(
                        1, len(generated_graphs) // len(target_entries)
                    )
                    graph_index = 0
                    for entry in target_entries:
                        section_graphs = generated_graphs[
                            graph_index : graph_index + graphs_per_section
                        ]
                        if section_graphs:
                            if "graphs" not in entry:
                                entry["graphs"] = []
                            entry["graphs"].extend(section_graphs)
                            logger.info(
                                f"Attached {len(section_graphs)} graphs to section: {entry.get('title', 'unknown')}"
                            )
                            graph_index += graphs_per_section
                            if graph_index >= len(generated_graphs):
                                break

                    # Attach any remaining graphs to the first target section
                    if graph_index < len(generated_graphs):
                        remaining = generated_graphs[graph_index:]
                        if target_entries and "graphs" not in target_entries[0]:
                            target_entries[0]["graphs"] = []
                        if target_entries:
                            target_entries[0]["graphs"].extend(remaining)
                            logger.info(
                                f"Attached {len(remaining)} remaining graphs to first section: {target_entries[0].get('title', 'unknown')}"
                            )
                else:
                    logger.error("No sections available to attach graphs to!")

        # 3. Join all text for uniqueness check and humanization.
        raw_full_text = "\n\n".join(all_text_parts)

        # 4. Optional humanization (section-by-section + aggregate text).
        if spec.humanizer_config.enabled:
            humanized_parts: List[str] = []
            for entry in all_sections_payload:
                content = entry.get("content", "")
                if content and content.strip():
                    humanized_content = rephrasy_service.humanize_text_or_fallback(
                        content, order_id=spec.assignment_id, llm=llm
                    )
                    entry["content"] = humanized_content
                    humanized_parts.append(humanized_content)
                else:
                    humanized_parts.append(content)
            full_text = (
                "\n\n".join(humanized_parts) if humanized_parts else raw_full_text
            )
        else:
            full_text = raw_full_text

        # 5. Generate smart title based on assignment content
        # Use a preview of the generated content to create an appropriate title
        content_preview = ""
        if all_text_parts:
            # Use first section or first 500 words as preview
            content_preview = all_text_parts[0][:500] if all_text_parts[0] else ""

        generated_title = await _generate_assignment_title(spec, content_preview)

        # 6. Generate DOCX with all sections.
        # `generate_formatted_document_tool` is a LangChain StructuredTool because of
        # the @tool decorator; we call its underlying function via `.func`.
        try:
            doc_meta = generate_formatted_document_tool.func(
                title=generated_title,
                sections=all_sections_payload,
                output_filename=output_filename,
                format_style=spec.format_style,
            )
        except Exception as doc_error:
            logger.error(f"Failed to generate document: {doc_error}", exc_info=True)
            raise RuntimeError(
                f"Document generation failed: {str(doc_error)}. "
                "Please check the logs for more details."
            ) from doc_error

        # 7. Uniqueness check.
        corpus = corpus_texts or []
        uniqueness: Optional[UniquenessResult] = None
        if corpus:
            try:
                uniqueness = checker.check_uniqueness(full_text, corpus)
            except Exception as uniqueness_error:
                logger.warning(
                    f"Uniqueness check failed: {uniqueness_error}", exc_info=True
                )
                logger.info("Continuing without uniqueness check results.")
                # Don't fail the entire generation if uniqueness check fails

        return AssignmentGenerationResult(
            doc_path=doc_meta.file_path,
            full_text=full_text,
            uniqueness=uniqueness,
        )

    return run
