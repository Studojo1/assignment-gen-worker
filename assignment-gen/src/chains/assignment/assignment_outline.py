"""
Project: Studojo Agents
File: src/chains/assignment/assignment_outline.py
Author: Rithul Kamesh
Created: 17 November 2025
Description: Outline generator (index) for assignments, separate from content.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from models.assignment import AssignmentSpec


def build_assignment_outline_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[AssignmentSpec, Dict[str, Any]]:
    """
    Build a chain that generates a JSON outline for the whole assignment.

    Input: AssignmentSpec
    Output: dict parsed from JSON with a top-level "sections" array.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You design high-quality academic assignment outlines.\n\n"
                    "You will NOT write full content; you only design the structure.\n"
                    "You must respond with STRICT JSON, no comments, no markdown, no extra text.\n"
                ),
            ),
            (
                "human",
                (
                    "Design an outline for the following assignment.\n\n"
                    "Assignment spec JSON (for internal use only, do NOT echo verbatim):\n"
                    "{assignment_spec_json}\n\n"
                    "Use this information:\n"
                    "- Assignment type: {assignment_type}\n"
                    "- Course: {course_name} (level: {level})\n"
                    "- Topic / description: {description}\n"
                    "- Target length: ~{length_pages} pages (~{target_words_total} words)\n"
                    "- Formatting style: {format_style}\n"
                    "- Tone: {tone}\n"
                    "- Web search allowed: {allow_web_search}\n"
                    "- Graphs/tables allowed: {allows_graphs}\n"
                    "- Graph requirements: {graph_requirements}\n"
                    "- Citation sources policy: {citation_sources_policy}\n\n"
                    "Your task:\n"
                    "1) Propose 3–7 main sections (excluding References), each with an ID,\n"
                    "   title, and target_words.\n"
                    "2) Each main section may have 0–5 subsections with IDs, titles, and\n"
                    "   target_words.\n"
                    "3) The sum of all target_words should be approximately "
                    "{target_words_total} (± 20%).\n"
                    "4) Make the outline specific to the topic, not generic. Section titles\n"
                    "   must clearly reflect the subject (e.g., 'Mathematical Foundations of "
                    "the Hadamard Gate' instead of 'Background').\n"
                    "5) If graphs/tables (e.g., a truth table) are requested, clearly mark the\n"
                    "   sections where they will appear using a 'notes' field.\n\n"
                    "Output format (STRICT JSON, no comments, no trailing commas):\n"
                    'Top-level object with a single key "sections", whose value is an array\n'
                    "of section objects. Each section object has:\n"
                    '- "id": string (e.g., "1")\n'
                    '- "title": string\n'
                    "- "
                    '"target_words": integer\n'
                    '- "subsections": array of subsection objects, each with "id", "title",\n'
                    '  and "target_words". You may also add an optional "notes" field.\n'
                    "Example shape (do NOT include this literally, just follow the structure):\n"
                    '  {{"sections": [{{"id": "1", "title": "Some title", "target_words": 800,\n'
                    '                 "subsections": [{{"id": "1.1", "title": "Sub", "target_words": 400}}]}}]}}\n'
                    "Rules:\n"
                    "- Return ONLY valid JSON. No markdown, no comments.\n"
                    "- Ensure the JSON is a SINGLE valid object. Do not output multiple objects.\n"
                    "- Check for balanced braces {{}} and brackets [].\n"
                ),
            ),
        ]
    )

    def to_model_inputs(spec: AssignmentSpec) -> Dict[str, Any]:
        # Approximate target words if not explicitly set.
        if spec.length_words is not None:
            target_words = spec.length_words
        elif spec.length_pages is not None:
            target_words = int(spec.length_pages * 300)
        else:
            target_words = 1500

        length_pages = (
            spec.length_pages
            if spec.length_pages is not None
            else round(target_words / 300.0, 1)
        )

        return {
            "assignment_spec_json": spec.model_dump_json(),
            "assignment_type": spec.assignment_type or "report",
            "course_name": spec.course_name or "Unknown course",
            "level": spec.level or "unspecified",
            "description": spec.description or "",
            "length_pages": length_pages,
            "target_words_total": target_words,
            "format_style": spec.format_style or "unspecified",
            "tone": spec.tone or "formal academic",
            "allow_web_search": spec.allow_web_search,
            "allows_graphs": spec.allows_graphs,
            "graph_requirements": spec.graph_requirements,
            "citation_sources_policy": spec.citation_sources_policy
            or "any credible source",
        }

    llm_chain = prompt | llm | RunnableLambda(lambda msg: msg.content)  # type: ignore[arg-type]

    def parse_json_output(raw: str) -> Dict[str, Any]:
        import json
        import re
        import logging
        
        logger = logging.getLogger(__name__)

        try:
            # Clean up the response
            cleaned = raw.strip()
            logger.info(f"Parsing outline JSON (length: {len(cleaned)} chars)")
            
            # Remove markdown code fences if present
            if cleaned.startswith("```"):
                # Find the end of code fence
                first_newline = cleaned.find("\n")
                if first_newline > 0:
                    cleaned = cleaned[first_newline:].strip()
                # Remove trailing code fence
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            
            # Try to fix trailing commas (common JSON error)
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Strategy 1: Try standard parsing
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Strategy 2: Try to find the first complete JSON object
                # This handles cases where the LLM outputs multiple objects or extra text
                logger.info("Standard parsing failed, attempting to extract first JSON object")
                brace_count = 0
                start_pos = cleaned.find('{')
                if start_pos >= 0:
                    for i in range(start_pos, len(cleaned)):
                        if cleaned[i] == '{':
                            brace_count += 1
                        elif cleaned[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                partial_json = cleaned[start_pos:i+1]
                                try:
                                    data = json.loads(partial_json)
                                    logger.info("Successfully extracted JSON object from partial response")
                                    break
                                except json.JSONDecodeError:
                                    pass
                    else:
                        raise  # Re-raise if no valid object found
                else:
                    raise

            if isinstance(data, dict) and "sections" in data:
                sections = data.get("sections", [])
                if not sections or len(sections) == 0:
                    logger.warning(f"Outline has empty sections array. Raw response preview: {raw[:500]}")
                else:
                    logger.info(f"Successfully parsed outline with {len(sections)} sections")
                return data
            else:
                logger.warning(f"Outline response missing 'sections' key. Keys found: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                # If we have a dict but no sections, maybe it's wrapped differently?
                # Just return it and let validation fail downstream if needed, or try to recover
                return data
                
        except Exception as e:
            logger.error(f"Failed to parse outline JSON: {e}")
            logger.error(f"Raw response preview (first 500 chars): {raw[:500]}")
            
            # Strategy 3: Regex/Stack-based fallback for sections array
            try:
                logger.info("Attempting to extract sections using stack-based fallback")
                # Find start of sections array
                match = re.search(r'"sections"\s*:\s*\[', raw)
                if match:
                    start_pos = match.end() - 1  # Point to the [
                    bracket_count = 0
                    in_string = False
                    escape_next = False
                    
                    for i in range(start_pos, len(raw)):
                        char = raw[i]
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
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found end of array
                                    sections_json = raw[start_pos:i+1]
                                    # Try to clean up
                                    sections_json = re.sub(r',(\s*[}\]])', r'\1', sections_json)
                                    # Aggressive cleanup: remove extra closing braces/brackets between objects
                                    # e.g. "}, }," -> "},"
                                    sections_json = re.sub(r'}\s*}+\s*,', '},', sections_json)
                                    sections_json = re.sub(r']\s*]+\s*,', '],', sections_json)
                                    # Also handle case at end of array
                                    sections_json = re.sub(r'}\s*}+\s*]', '}]', sections_json)
                                    
                                    try:
                                        sections = json.loads(sections_json)
                                        if isinstance(sections, list):
                                            # Check if there are more objects after the array closes
                                            # Handle case where array closes early: ]},{"id":"3",...}
                                            remaining_text = raw[i+1:].strip()
                                            if remaining_text.startswith(',') or remaining_text.startswith('},{"'):
                                                # Extract additional objects that appear after array close
                                                logger.info("Detected objects outside array, attempting to extract...")
                                                remaining_cleaned = remaining_text.lstrip(',').strip()
                                                # Try to extract additional objects
                                                additional_objects = []
                                                remaining_text_with_brackets = remaining_cleaned
                                                # Handle case where objects start with },{ or just {
                                                if remaining_text_with_brackets.startswith('},{'):
                                                    remaining_text_with_brackets = remaining_text_with_brackets[1:]  # Remove leading ,
                                                elif remaining_text_with_brackets.startswith('{'):
                                                    pass  # Already starts with {
                                                else:
                                                    remaining_text_with_brackets = remaining_text_with_brackets.lstrip(',').strip()
                                                
                                                # Extract objects while handling string escaping
                                                brace_start = remaining_text_with_brackets.find('{')
                                                while brace_start >= 0:
                                                    brace_count = 0
                                                    in_string = False
                                                    escape_next = False
                                                    obj_start = brace_start
                                                    
                                                    for j in range(obj_start, len(remaining_text_with_brackets)):
                                                        char = remaining_text_with_brackets[j]
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
                                                            if char == '{':
                                                                brace_count += 1
                                                            elif char == '}':
                                                                brace_count -= 1
                                                                if brace_count == 0:
                                                                    obj_json = remaining_text_with_brackets[obj_start:j+1]
                                                                    try:
                                                                        obj = json.loads(obj_json)
                                                                        additional_objects.append(obj)
                                                                        logger.info(f"Extracted additional section: {obj.get('id', 'unknown')}")
                                                                    except Exception as parse_err:
                                                                        logger.warning(f"Failed to parse additional object: {parse_err}")
                                                                    remaining_text_with_brackets = remaining_text_with_brackets[j+1:].lstrip(',').strip()
                                                                    brace_start = remaining_text_with_brackets.find('{')
                                                                    break
                                                    else:
                                                        break  # No complete object found
                                                
                                                if additional_objects:
                                                    sections.extend(additional_objects)
                                                    logger.info(f"Added {len(additional_objects)} additional sections, total: {len(sections)}")
                                            
                                            logger.info(f"Successfully extracted {len(sections)} sections via stack extraction")
                                            return {"sections": sections}
                                    except Exception as parse_err:
                                        logger.warning(f"Failed to parse extracted sections JSON: {parse_err}")
                                    break
            except Exception as extract_error:
                logger.error(f"Failed to extract sections with stack method: {extract_error}")
        
        # Final Fallback: wrap raw string so caller can debug.
        logger.error("Returning empty outline due to parsing failure")
        return {"sections": [], "raw": raw, "parse_error": True}

    return (
        RunnableLambda(to_model_inputs) | llm_chain | RunnableLambda(parse_json_output)
    )
