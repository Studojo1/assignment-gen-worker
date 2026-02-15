"""
Project: Studojo Agents
File: src/chains/assignment/assignment_input.py
Author: Rithul Kamesh
Created: 16 November 2025
Description: LangChain-based Assignment Input Collector (Step 3 in `doc/flow.md`).

This chain behaves like a small state machine:
- It looks at the current AssignmentSpec.
- It reads the latest user message.
- It decides what to ask next, or whether the spec is "ready".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from pydantic import BaseModel

from models.assignment import AssignmentSpec, AssignmentStatus
from tools.common.web import search_and_fetch
from .search_utils import (
    build_legacy_search_query,
    extract_search_topic,
    generate_search_query_with_llm,
)
from utils.resource_analyzer import ResourceAnalyzer

logger = logging.getLogger(__name__)


class AssignmentInputCollectorOutput(BaseModel):
    """Structured output from the assignment input collector."""

    assistant_message: str
    assignment_spec: AssignmentSpec
    assignment_spec_updates: Dict[str, Any]
    is_complete: bool
    web_search_results: Optional[List[Dict[str, Any]]] = None


def _apply_assignment_spec_updates(
    spec: AssignmentSpec, updates: Dict[str, Any]
) -> AssignmentSpec:
    """Merge validated updates into the current assignment spec."""
    if not updates:
        return spec.model_copy()

    merged = spec.model_dump()
    merged.update(updates)
    try:
        return AssignmentSpec.model_validate(merged)
    except Exception:
        logger.exception("Failed to apply assignment spec updates, returning original spec")
        return spec.model_copy()


def _build_collector_output(
    spec: AssignmentSpec, parsed: Dict[str, Any]
) -> AssignmentInputCollectorOutput:
    """Produce final collector output including updated spec and sources."""
    updated_spec = _apply_assignment_spec_updates(
        spec, parsed.get("assignment_spec_updates", {})
    )
    if parsed.get("is_complete"):
        updated_spec.status = AssignmentStatus.READY

    return AssignmentInputCollectorOutput(
        assistant_message=parsed.get("assistant_message", ""),
        assignment_spec=updated_spec,
        assignment_spec_updates=parsed.get("assignment_spec_updates", {}),
        is_complete=bool(parsed.get("is_complete", False)),
        web_search_results=updated_spec.web_search_results,
    )


def build_assignment_input_collector_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[Dict[str, Any], AssignmentInputCollectorOutput]:
    """
    Build a chain that collects/updates an AssignmentSpec over multiple turns.

    Input: dict with
      - user_message: str
      - assignment_spec: dict (serialized AssignmentSpec)

    Output: AssignmentInputCollectorOutput
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant helping a student define an assignment spec.\n"
                    "You are given the current assignment spec as JSON and the user's "
                    "latest message.\n\n"
                    "CRITICAL RULES - Follow these strictly:\n"
                    "1) INFER FIRST, ASK LATER: If you can reasonably infer ANY field value "
                    "from the user's message or existing spec context, DO NOT ask about it. "
                    "However, if a file is provided but key constraints (like word count, specific focus, "
                    "or format) are missing and not inferable, YOU MUST ASK about them. Do not assume "
                    "defaults if the user's request implies a specific academic standard.\n"
                    "2) RELAXED MANDATORY FIELDS: You should try to get 'length' and 'format_style', but DO NOT BLOCK "
                    "progress if they are missing. Instead, PROPOSE reasonable defaults (e.g., 'I'll assume 2-3 pages and APA format "
                    "unless you prefer otherwise') and move on to the actual assignment content. \n"
                    "3) CONTEXT EXTRACTION IS PRIORITY: Your primary goal is to extract assignment details "
                    "from the uploaded file context. Look for ANY relevant details: specific questions, "
                    "topics, grading criteria, formatting rules, or constraints. \n"
                    "   - If the user asks for 'CIA 1' or 'Assignment 2', look for a table row, a section header, "
                    "     or a list item matching that name. \n"
                    "   - If you find a matching row/section, EXTRACT ALL DETAILS from it (topic, due date, "
                    "     specific instructions) and use them to populate the spec.\n"
                    "   - Do NOT ask 'What is the topic for CIA 1?' if the file clearly says 'CIA 1: Case Study on X'.\n"
                    "4) LIMIT QUESTIONS: Ask AT MOST 1-2 questions per turn. Batch related "
                    "questions together if you must ask multiple things.\n"
                    "5) AVOID REDUNDANCY: Never ask about fields that are already set in "
                    "the spec unless the user explicitly contradicts them.\n"
                    "6) STOP WHEN READY: Once you have the essential information (including length/format/topic), mark "
                    "is_complete=true immediately. Do not ask confirmatory questions.\n"
                    "7) SMART INFERENCE: Use context clues - if the user mentions 'IEEE format', "
                    "set format_style. If they say '5 pages', set length_pages. If they mention "
                    "a course name, set course_name. Be proactive, not passive.\n\n"
                    "8) NARRATIVE FIDELITY: Treat uploaded narrative text as the source of truth. "
                    "Explicitly extract geography, sector focus, constraints, and framing from that text "
                    "before falling back to generic assumptions.\n"
                    "9) LOCATION LOCK: If the narrative or user explicitly mentions India (or any "
                    "other geography), bake that exact locale into description/structure requirements "
                    "and avoid defaulting to U.S.-centric examples unless the user overrides it.\n"
                    "10) POLICY STRUCTURE: When the user wants policy recommendations, note in "
                    "structure_requirements that recommendations must appear as clear Problem -> Solution "
                    "pairs tailored to the specified geography.\n"
                    "11) COMPLEX ASSIGNMENT PARSING: \n"
                    "    - STRUCTURE MAPPING: If the file contains a 'Final Submission', 'Report Structure', or 'Deliverables' "
                    "      section, map those items directly to 'structure_requirements'. Ensure the order matches the file.\n"
                    "    - VARIABLE IDENTIFICATION (CRITICAL): If the text says 'Select X', 'Choose Y', 'Pick one' (e.g., 'Select one listed company', "
                    "      'Choose 5 terminologies'), you MUST identify this as a user choice. \n"
                    "      - ASK THE USER: 'The assignment asks you to select a company. Which one would you like to focus on?'\n"
                    "      - Do NOT pick for them unless asked. This is more important than asking about format/length.\n"
                    "    - MISSING ATTACHMENTS: If the text refers to an 'attached list', 'provided spreadsheet', or 'appendix' "
                    "      that is NOT in the extracted text, explicitly ask the user to provide it or paste the content.\n"
                    "12) FORCE COMPLETION CAREFULLY: If the user explicitly says 'stop asking', 'just generate', "
                    "'I am done', or 'no more questions', you MUST set is_complete=true IMMEDIATELY. "
                    "However, if the user says 'Complete X' or 'Do this assignment', treat it as a goal, "
                    "NOT a command to stop questioning. If critical info (length/format/topic) is missing, ASK about it. "
                    "Only force completion if the spec is truly ready or the user explicitly refuses to answer.\n\n"
                    "Your job each turn:\n"
                    "1) Update the spec with ALL information you can infer from the user message.\n"
                    "2) CHECK FOR EXIT: If the user explicitly wants to stop/generate (e.g. 'just do it'), set is_complete=true.\n"
                    "3) PRIORITY: If web search results are provided (especially if they were just found), "
                    "you SHOULD ask the user relevant questions about them. Examples:\n"
                    "   - 'I found these sources about [topic]. Are these appropriate for your assignment?'\n"
                    "   - 'Should I focus on any particular aspect from these sources?'\n"
                    "   - 'Do you want me to emphasize any specific angle or perspective from these sources?'\n"
                    "   - 'Are there any additional sources or topics you'd like me to search for?'\n"
                    "   When web search results are fresh (just found), asking about them takes priority "
                    "over other questions.\n"
                    "4) Ask ONLY 1-2 essential questions if critical information is still missing.\n"
                    "5) If the spec has enough information to proceed OR the user forced completion, mark is_complete=true.\n\n"
                    "An assignment spec is considered COMPLETE when it has:\n"
                    "- assignment_type (or can infer from context)\n"
                    "- description (the main topic/prompt - MUST BE SPECIFIC)\n"
                    "- length (words OR pages - MANDATORY)\n"
                    "- format_style (MANDATORY)\n"
                    "- allow_web_search (true/false - ask only if not inferable)\n"
                    "- allows_graphs (true/false - MUST ASK USER TO CONFIRM before setting)\n\n"
                    "CRITICAL: Graph handling rules:\n"
                    "- If the assignment description mentions graphs/charts/visualizations/figures/plots, "
                    "you MUST ask the user to confirm if they want graphs generated.\n"
                    "- DO NOT automatically set allows_graphs=true just because keywords are present.\n"
                    "- Ask: 'I noticed your assignment mentions [graph/chart/visualization]. Would you like me to include graphs in your assignment?'\n"
                    "- Only set allows_graphs=true after the user explicitly confirms they want graphs.\n\n"
                    "OPTIONAL fields (do NOT require these to mark complete):\n"
                    "- course_name (nice to have but not required)\n"
                    "- level (can infer or leave unspecified)\n"
                    "- format_style (can infer from context or use default)\n"
                    "- tone (can infer from context or use default)\n"
                    "- due_date (truly optional - if provided, use ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)\n"
                    "- graph_requirements (only if allows_graphs=true)\n\n"
                    "Web search handling:\n"
                    "- If web_search_results are provided in the context (especially if marked as 'just found'), "
                    "you MUST ask relevant questions about them. This is a priority when results are fresh.\n"
                    "- Ask about: relevance of sources, specific aspects to focus on, additional sources needed, "
                    "particular angles or perspectives to emphasize.\n"
                    "- Do NOT ask redundant questions about web search if you've already asked about the same "
                    "results in a previous turn, unless the user's response indicates they want to discuss them further.\n\n"
                    "10) OUTPUT FORMAT: You are a STATE UPDATE MACHINE. You must ONLY return changes.\n"
                    "NEVER return the full assignment spec. NEVER return the 'assignment_spec' key.\n"
                    "Return a JSON object with exactly these keys:\n"
                    "- assistant_message: string (your response to the user)\n"
                    "- assignment_spec_updates: object (key-value pairs of fields to UPDATE)\n"
                    "- is_complete: boolean (true if ready to generate)\n\n"
                    "11) RESOURCE AWARENESS: You will be provided with a 'resource_analysis_context'. "
                    "Use this to ask specific, intelligent questions based on the uploaded files and web sources. "
                    "Prioritize these questions over generic ones.\n\n"
                    "Example (CORRECT - asking about graphs):\n"
                    '{{"assistant_message": "I noticed your assignment mentions graphs. Would you like me to include graphs in your assignment?", "assignment_spec_updates": {{"description": "Report on economic indicators"}}, "is_complete": false}}\n'
                    "Example (INCORRECT - don't auto-set allows_graphs):\n"
                    '{{"assistant_message": "I will create graphs for Real GDP.", "assignment_spec_updates": {{"description": "Report on economic indicators", "allows_graphs": true}}, "is_complete": false}}'
                ),
            ),
            (
                "human",
                (
                    "Current state summary:\n"
                    "{current_state_summary}\n\n"
                    "{resource_analysis_context}\n"
                    "{uploaded_files_context}\n"
                    "{web_search_context}\n"
                    "User message:\n"
                    "{user_message}\n\n"
                    "REMINDER: If the user says 'generate', 'ready', 'done', or similar, you MUST set is_complete=true. "
                    "Do not ask for more details. Use reasonable defaults for any missing fields."
                ),
            ),
        ]
    )

    async def to_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model inputs: serialize AssignmentSpec to JSON string and perform web search if needed."""
        spec: AssignmentSpec = inputs["assignment_spec"]
        user_message: str = inputs["user_message"]

        # Include uploaded files context if available
        uploaded_files_context = ""
        if spec.uploaded_files and len(spec.uploaded_files) > 0:
            file_parts = []
            for i, file_info in enumerate(spec.uploaded_files, 1):
                filename = file_info.get("filename", f"File {i}")
                extracted_text = file_info.get("extracted_text", "")
                if extracted_text:
                    # Show preview of file content (increased to 50k chars to capture full course plans)
                    preview = extracted_text[:50000]
                    file_parts.append(
                        f"=== Uploaded File {i}: {filename} ===\n"
                        f"Content:\n{preview}"
                        f"{'...' if len(extracted_text) > 50000 else ''}\n"
                        f"Total length: {len(extracted_text)} characters\n"
                    )
            if file_parts:
                uploaded_files_context = (
                    "\n=== UPLOADED FILES AVAILABLE ===\n"
                    "The user has uploaded the following file(s). Use this content to help "
                    "understand the assignment requirements, course context, and any specific "
                    "instructions. Extract relevant information like course name, assignment type, "
                    "requirements, due dates, etc. from these files:\n\n"
                    + "\n".join(file_parts)
                    + "\n"
                )

        # Perform web search if we have description and web search is enabled/allowed
        # Do this BEFORE marking complete, so we can ask questions about sources
        web_search_context = ""
        web_search_performed = False

        has_substantial_description = bool(
            spec.description and spec.description.strip()
        )
        user_rejected_web_search = any(
            phrase in user_message.lower()
            for phrase in ["no web", "don't search", "no search", "without web"]
        )

        # Auto-enable web search if user mentions it
        user_enabled_web_search = any(
            phrase in user_message.lower()
            for phrase in [
                "use web search",
                "web search",
                "search the web",
                "you can use web",
                "can use web search",
            ]
        )

        # If user explicitly enables web search, set it
        if user_enabled_web_search and spec.allow_web_search is None:
            spec.allow_web_search = True
            logger.info("Auto-enabled web search based on user message")

        # Use LLM to generate search query
        search_query = ""
        
        # Use description from spec, or fallback to user_message if spec is empty (first turn)
        effective_description = spec.description or ""
        if not effective_description and len(user_message) > 50: # Heuristic for "substantial" message
             effective_description = user_message

        if effective_description or uploaded_files_context:
            try:
                search_query = await generate_search_query_with_llm(
                    llm,
                    effective_description,
                    spec.course_name,
                    uploaded_files_context if uploaded_files_context else None,
                )
                if search_query:
                    logger.info(f"LLM generated search query: {search_query}")
            except Exception as e:
                logger.error(f"Failed to generate search query with LLM: {e}")

        # Fallback to legacy if LLM fails or returns empty
        if not search_query and spec.description:
            search_query = build_legacy_search_query(spec.description, spec.course_name)
            if search_query:
                logger.info(
                    "LLM returned empty query; falling back to legacy heuristics."
                )

        should_search = (
            bool(search_query)
            and spec.web_search_results is None
            and spec.allow_web_search is not False  # Only skip if explicitly False
            and not user_rejected_web_search
        )

        # Debug logging for web search decision
        logger.debug(f"Web search decision: should_search={should_search}")
        logger.debug(f"  - has_substantial_description: {has_substantial_description}")
        logger.debug(
            f"  - web_search_results is None: {spec.web_search_results is None}"
        )
        logger.debug(f"  - allow_web_search: {spec.allow_web_search}")
        logger.debug(f"  - user_rejected_web_search: {user_rejected_web_search}")
        if spec.description:
            logger.debug(
                f"  - description length: {len(spec.description.split())} words"
            )
            logger.debug(f"  - description preview: {spec.description[:100]}...")

        if should_search:
            try:
                logger.info(
                    "Performing web search during input collection: %s", search_query
                )
                if spec.description:
                    logger.debug(
                        "Search query constructed from description (preview: %s...)",
                        spec.description[:100],
                    )
                web_results = await search_and_fetch(search_query, max_results=5)
                logger.info(f"Web search returned {len(web_results)} raw results")

                if web_results:
                    # Filter for relevance (less aggressive)
                    filtered_results = []
                    query_lower = search_query.lower()
                    # Extract meaningful query terms (exclude common words)
                    common_words = {
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "but",
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                        "of",
                        "with",
                        "by",
                    }
                    query_terms = set(
                        w
                        for w in query_lower.split()
                        if w not in common_words and len(w) > 2
                    )

                    logger.debug(
                        f"Filtering {len(web_results)} results using query terms: {query_terms}"
                    )

                    for result in web_results:
                        title = result.get("title", "").lower()
                        snippet = result.get("snippet", "").lower()
                        url = result.get("url", "").lower()
                        text_to_check = f"{title} {snippet}"

                        # Calculate relevance score (more lenient)
                        relevance_score = sum(
                            1 for term in query_terms if term in text_to_check
                        )

                        # Filter out obviously irrelevant results (spam/off-topic)
                        irrelevant_keywords = [
                            "richest",
                            "jigsaw",
                            "toilet",
                            "diy.stackexchange",
                            "answers.com",
                            "quora.com/personal",
                            "reddit.com/r/random",
                        ]
                        is_irrelevant = any(
                            kw in text_to_check or kw in url
                            for kw in irrelevant_keywords
                        )

                        # More lenient filtering: accept if relevance_score > 0 OR if it's not obviously irrelevant
                        # For academic topics, even partial matches can be useful
                        if (
                            relevance_score > 0 or len(query_terms) == 0
                        ) and not is_irrelevant:
                            filtered_results.append(result)
                            logger.debug(
                                f"Accepted result: {result.get('title', 'Unknown')[:50]}... (score: {relevance_score})"
                            )
                        else:
                            logger.debug(
                                f"Filtered out result: {result.get('title', 'Unknown')[:50]}... (score: {relevance_score}, irrelevant: {is_irrelevant})"
                            )

                    # Fallback: if all results filtered out, use top 3 anyway (they might still be useful)
                    if not filtered_results and web_results:
                        logger.warning(
                            "All results filtered out, using top 3 results as fallback"
                        )
                        filtered_results = web_results[:3]

                    if filtered_results:
                        # Store results in spec (will be updated by LLM)
                        spec.web_search_results = filtered_results[:5]
                        web_search_performed = True

                        # Build context string for prompt with clear indicators for asking questions
                        web_parts = []
                        for i, result in enumerate(filtered_results[:5], 1):
                            title = result.get("title", "Unknown Title")
                            url = result.get("url", "")
                            snippet = result.get("snippet", "")
                            web_parts.append(
                                f"[{i}] {title}\n   URL: {url}\n   Summary: {snippet}"
                            )
                        web_search_context = (
                            "=== WEB SEARCH RESULTS (JUST FOUND) - ASK USER QUESTIONS ABOUT THESE ===\n"
                            f"I found {len(filtered_results)} relevant sources about the assignment topic. "
                            "You SHOULD ask the user relevant questions about these sources, such as:\n"
                            "- Are these sources appropriate for the assignment?\n"
                            "- Should I focus on any particular aspects from these sources?\n"
                            "- Are there specific angles or perspectives to emphasize?\n"
                            "- Do you need any additional sources or topics searched?\n\n"
                            "Sources found:\n" + "\n\n".join(web_parts) + "\n\n"
                        )
                        logger.info(
                            f"Found {len(filtered_results)} relevant web sources (from {len(web_results)} raw results)"
                        )
                    else:
                        logger.warning(
                            f"No results passed filtering (had {len(web_results)} raw results)"
                        )
                else:
                    logger.warning(
                        f"Web search returned no results for query: {search_query}"
                    )
            except Exception as e:
                logger.error(
                    f"Web search during input collection failed: {e}", exc_info=True
                )
                logger.warning(
                    "Continuing without web sources. You can retry search later if needed."
                )

        # If web search results already exist, include them in context
        if not web_search_performed and spec.web_search_results:
            web_parts = []
            for i, result in enumerate(spec.web_search_results[:5], 1):
                title = result.get("title", "Unknown Title")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                web_parts.append(f"[{i}] {title}\n   URL: {url}\n   Summary: {snippet}")
            web_search_context = (
                "=== PREVIOUSLY FOUND WEB SEARCH RESULTS ===\n"
                "These sources were found earlier. You may reference them or ask follow-up questions if relevant:\n\n"
                + "\n\n".join(web_parts)
                + "\n\n"
            )

        if not web_search_context:
            web_search_context = ""

        if not uploaded_files_context:
            uploaded_files_context = ""

        # Create a lightweight version of spec for the prompt
        # We strip out heavy content that is already provided in context variables
        # This prevents the LLM from seeing (and trying to echo) massive JSON objects
        spec_for_prompt = spec.model_copy()
        spec_for_prompt.uploaded_files = []  # Provided in uploaded_files_context
        spec_for_prompt.web_search_results = []  # Provided in web_search_context

        # Run resource analysis
        resource_analysis_context = ResourceAnalyzer.analyze_resources(spec)

        return {
            "current_state_summary": spec_for_prompt.model_dump_json(),
            "user_message": user_message,
            "uploaded_files_context": uploaded_files_context,
            "web_search_context": web_search_context,
            "resource_analysis_context": resource_analysis_context,
        }

    llm_chain = prompt | llm | RunnableLambda(lambda msg: msg.content)  # type: ignore[arg-type]

    def parse_output(raw: str) -> Dict[str, Any]:
        """Parse the JSON string from the model into structured data."""
        import json
        import re

        # Clean the raw string - remove markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Try to extract JSON if it's wrapped in other text
        # Look for JSON object pattern - must have quoted keys to be valid JSON
        json_match = re.search(r'\{\s*"[^"]+"\s*:', cleaned, re.DOTALL)
        if json_match:
            # Found start of JSON object, now find the matching closing brace
            start_pos = json_match.start()
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(start_pos, len(cleaned)):
                char = cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            # Found matching closing brace
                            cleaned = cleaned[start_pos : i + 1]
                            break

        try:
            # Try to fix common JSON issues before parsing
            # Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            error_pos = getattr(e, "pos", None)
            # Log the actual response for debugging
            logger.error(f"Failed to parse JSON response: {e}")
            if error_pos:
                logger.error(f"JSON error at position {error_pos} (char {error_pos})")
                logger.error(
                    f"Context around error: {cleaned[max(0, error_pos-50):error_pos+50]}"
                )
            logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
            logger.debug(f"Cleaned response (first 500 chars): {cleaned[:500]}")
            logger.debug(f"Cleaned response (last 200 chars): {cleaned[-200:]}")

            # Try multiple strategies to extract partial information
            data = None

            # Strategy 1: Try to extract complete JSON object by finding matching braces
            try:
                brace_count = 0
                start_pos = cleaned.find("{")
                if start_pos >= 0:
                    for i in range(start_pos, len(cleaned)):
                        if cleaned[i] == "{":
                            brace_count += 1
                        elif cleaned[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                partial_json = cleaned[start_pos : i + 1]
                                try:
                                    data = json.loads(partial_json)
                                    logger.info(
                                        "Successfully extracted complete JSON object from partial response"
                                    )
                                    break
                                except json.JSONDecodeError:
                                    pass
            except Exception as extract_error:
                logger.debug(f"Strategy 1 (brace matching) failed: {extract_error}")

            # Strategy 2: Try to extract assistant_message field specifically
            if data is None:
                try:
                    msg_match = re.search(
                        r'"assistant_message"\s*:\s*"((?:[^"\\]|\\.)*)"',
                        cleaned,
                        re.DOTALL,
                    )
                    if msg_match:
                        assistant_msg = msg_match.group(1)
                        # Unescape JSON string
                        try:
                            assistant_msg = assistant_msg.encode().decode(
                                "unicode_escape"
                            )
                        except (UnicodeDecodeError, UnicodeError):
                            pass
                        logger.info(
                            "Extracted partial assistant_message from malformed JSON"
                        )
                        return {
                            "assistant_message": assistant_msg
                            + " (Note: There was an issue parsing the full response. Please try rephrasing your request.)",
                            "assignment_spec_updates": {},
                            "is_complete": False,
                        }
                except Exception as extract_error:
                    logger.debug(
                        f"Strategy 2 (message extraction) failed: {extract_error}"
                    )

            # If we still don't have data, use fallback
            if data is None:
                # Extremely defensive fallback: provide helpful error message
                # Log the issue for debugging but give user actionable feedback
                logger.error(
                    f"Failed to parse LLM response as JSON. "
                    f"This usually means the model returned an unexpected format. "
                    f"Raw response length: {len(raw)} chars"
                )

                # Try to provide a helpful message based on what we can see
                if len(raw) < 50:
                    error_msg = (
                        "I received a very short response that I couldn't understand. "
                        "Could you please rephrase your request? Try breaking it into smaller parts if it's complex."
                    )
                elif "error" in raw.lower() or "sorry" in raw.lower():
                    error_msg = (
                        "The system encountered an issue processing your request. "
                        "Please try rephrasing it or breaking it into smaller parts. "
                        "You can also try: /spec to see current status, or /reset to start over."
                    )
                else:
                    error_msg = (
                        "I had trouble understanding your request in the format I expected. "
                        "This can happen with very long or complex requests. "
                        "Please try:\n"
                        "  - Rephrasing your request more simply\n"
                        "  - Breaking it into smaller parts\n"
                        "  - Using /spec to check current status\n"
                        "  - Using /reset to start fresh if needed"
                    )

                return {
                    "assistant_message": error_msg,
                    "assignment_spec_updates": {},
                    "is_complete": False,
                }

        # Handle assignment_spec_updates
        updates = data.get("assignment_spec_updates", {})
        # Fallback for backward compatibility if LLM still returns assignment_spec
        if not updates and "assignment_spec" in data:
            updates = data["assignment_spec"]

        # Parse due_date if it's a string
        if "due_date" in updates:
            if not updates["due_date"]:
                # Handle empty string or null
                updates["due_date"] = None
            elif isinstance(updates["due_date"], str):
                due_date_value = updates["due_date"]
                try:
                    # Try using dateutil parser for flexible date parsing
                    from dateutil import parser as date_parser

                    updates["due_date"] = date_parser.parse(due_date_value)
                except (ImportError, ValueError, TypeError):
                    # Fallback: try common formats manually
                    try:
                        from datetime import datetime

                        # Try ISO format first
                        updates["due_date"] = datetime.fromisoformat(
                            due_date_value.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        # If all parsing fails, log and set to None
                        logger.warning(
                            f"Could not parse due_date '{due_date_value}'. "
                            "Setting to None. Please use ISO format (YYYY-MM-DD) or standard date formats."
                        )
                        updates["due_date"] = None

        # Normalize status if present
        if "status" in updates and isinstance(updates["status"], str):
            updates["status"] = updates["status"].lower()

        assistant_msg = data.get(
            "assistant_message",
            "Okay, let's keep refining your assignment details.",
        )
        is_complete = bool(data.get("is_complete", False))

        # Heuristic: If the LLM says it's complete/ready in the text but forgot to set the flag,
        # we force it to True. This handles cases where the LLM's "speech" and "action" diverge.
        lower_msg = assistant_msg.lower()
        if not is_complete and (
            "ready to generate" in lower_msg
            or "spec is complete" in lower_msg
            or "assignment spec is now complete" in lower_msg
            or "no further questions" in lower_msg
        ):
            logger.info("Forcing is_complete=True based on assistant message content.")
            is_complete = True

        return {
            "assistant_message": assistant_msg,
            "assignment_spec_updates": updates,
            "is_complete": is_complete,
        }

    async def run_chain(
        inputs: Dict[str, Any]
    ) -> AssignmentInputCollectorOutput:
        """Execute the collector chain end-to-end."""
        spec: AssignmentSpec = inputs["assignment_spec"]
        prompt_inputs = await to_model_inputs(inputs)
        raw_output = await llm_chain.ainvoke(prompt_inputs)
        parsed = parse_output(raw_output)
        return _build_collector_output(spec, parsed)

    return RunnableLambda(run_chain)
