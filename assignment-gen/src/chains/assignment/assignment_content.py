"""
Project: Studojo Agents
File: src/chains/assignment/assignment_content.py
Author: Rithul Kamesh
Created: 17 November 2025
Description: Section-level content generator chain for assignments (Step 6).

This chain wraps the long prompt in `doc/flow.md` that:
- Generates academically styled content for a main section and its subsections.
- Returns ONLY JSON mapping 'main' + subsection keys to plain text.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable


class SubsectionSpec(TypedDict):
    key: str
    title: str
    pages: float


def _build_prompt() -> ChatPromptTemplate:
    """Return the ChatPromptTemplate encoding the section-generation instructions."""

    template = (
        "Return only valid JSON. Do not include code fences or commentary. "
        "Escape newlines and tabs in JSON strings (use \\n and \\t).\n\n"
        "Generate a detailed, well-structured, and academically sound draft for a "
        "specific subsection of an assignment. The content must align with the "
        "overall assignment description, meet rubric expectations, reflect course "
        "objectives/outcomes, and fit within the provided page/word limits. The "
        "output must be written in formal, scholarly prose (no instructions, no "
        "meta-commentary about assignments, no tutorial-style phrasing). The text "
        "should read as if it is part of a completed academic report written for "
        "submission.\n\n"
        "You will generate content for a main section and all its subsections "
        "together, using the context below. Produce a JSON object mapping each "
        "subsection key to a plain-text paragraph block.\n\n"
        "Context:\n"
        "- Section: {section_number} {section_title}\n"
        "- Target total pages for this section (approx): {target_pages}\n"
        "- Target words for this section (approx): {target_words_total}\n\n"
        "Assignment description:\n"
        "{courseplan_analysis}\n\n"
        "Student responses:\n"
        "{user_responses}\n\n"
        "Graph context (may be empty):\n"
        "{graph_context_block}\n\n"
        "Web search results (may be empty; cite using the specified format if used):\n"
        "{web_search_block}\n\n"
        "Excel data insights (may be empty; do not fabricate):\n"
        "{excel_context_snippet}\n\n"
        "{citation_instruction}\n\n"
        "Subsections to fill (keys must match exactly in the JSON output):\n"
        "{subsections_json}\n\n"
        "CRITICAL REQUIREMENTS FOR JSON OUTPUT:\n"
        "1. You MUST include ALL keys from the subsections list above in your JSON output.\n"
        "2. The subsection keys in your JSON MUST exactly match the 'key' field from the subsections list.\n"
        "3. For example, if a subsection has key '1_1', your JSON must include '1_1' as a key (NOT '1.1', '1-1', or any other variation).\n"
        "4. Missing keys will result in empty sections in the final document.\n"
        "5. Your JSON must be complete and valid - include ALL required keys.\n\n"
        "JSON Output Format:\n"
        "- Return ONLY valid JSON (no markdown code fences, no commentary).\n"
        "- Your JSON must have this exact structure:\n"
        "  {{\n"
        '    "main": "<content for main section, plain text>",\n'
        '    "1_1": "<content for subsection 1.1, plain text>",\n'
        '    "1_2": "<content for subsection 1.2, plain text>",\n'
        "    ... one key for EACH subsection key from the list above ...\n"
        "  }}\n"
        "- Every key from the subsections list MUST appear in your JSON output.\n"
        "- If the subsections list shows keys ['1_1', '1_2', '2_1'], your JSON MUST include all three keys.\n"
        "- No markdown, no structural markers; paragraphs separated by blank lines.\n"
        "- You MUST treat the target word count as a hard constraint, not a vague hint.\n"
        '- Aim for approximately {target_words_total} words TOTAL across "main" plus\n'
        "  all subsection keys.\n"
        "- Never produce fewer than 0.8 × {target_words_total} words unless the input\n"
        "  context is extremely short.\n"
        "- If the context is short but the target is large, expand with concrete\n"
        "  explanations, examples, comparisons, implications, and limitations that are\n"
        "  specifically about this topic.\n"
        "- It is better to slightly exceed the target than to fall more than 20% below it.\n"
        "- Avoid repetition, boilerplate, or generic filler sentences; every sentence\n"
        "  must contribute something specific and contentful.\n"
        "- Use at least one concrete detail from the context when relevant.\n"
        "- Do not describe the purpose of sections, the assignment, or references. "
        "Write only the content itself, as if for final submission.\n\n"
        "Mathematical Equations:\n"
        "- For mathematical expressions, use LaTeX syntax that will be properly formatted in the final document.\n"
        "- Use $equation$ for inline equations (e.g., The Hadamard gate is $H = \\frac{{1}}{{\\sqrt{{2}}}} \\begin{{pmatrix}} 1 & 1 \\\\ 1 & -1 \\end{{pmatrix}}$).\n"
        "- Use $$equation$$ for block/display equations that should be centered on their own line.\n"
        "- Common notations:\n"
        "  * Subscripts: $x_i$, $H_{{gate}}$, $\\alpha_1$\n"
        "  * Superscripts: $x^2$, $e^{{-i\\theta}}$, $x^{{n+1}}$\n"
        "  * Square roots: $\\sqrt{{x}}$, $\\sqrt[{{n}}]{{x}}$, $\\sqrt{{x^2 + y^2}}$\n"
        "  * Fractions: $\\frac{{a}}{{b}}$, $\\frac{{1}}{{2}}$, $\\frac{{numerator}}{{denominator}}$\n"
        "  * Matrices: $\\begin{{pmatrix}} a & b \\\\ c & d \\end{{pmatrix}}$\n"
        "  * Integrals: $\\int_{{0}}^{{1}} x dx$, $\\iint_D f(x,y) dx dy$\n"
        "  * Sums/products: $\\sum_{{i=1}}^{{n}} x_i$, $\\prod_{{i=1}}^{{n}} a_i$\n"
        "  * Greek letters: $\\alpha$, $\\beta$, $\\theta$, $\\pi$, $\\Delta$, $\\Omega$\n"
        "  * Operators: $\\pm$, $\\mp$, $\\times$, $\\cdot$, $\\div$\n"
        "  * Relations: $\\leq$, $\\geq$, $\\neq$, $\\approx$, $\\equiv$\n"
        "  * Set notation: $\\in$, $\\subset$, $\\cup$, $\\cap$, $\\emptyset$\n"
        "  * Other: $\\infty$, $\\partial$, $\\nabla$, $\\forall$, $\\exists$\n"
        "- Equations will be properly rendered with subscripts, superscripts, square roots, "
        "and all mathematical notation in the final document.\n\n"
        "Task Instructions for the Agent (do NOT mention these explicitly in output):\n"
        "1. Analyze the assignment context and rubric internally; do not echo this.\n"
        "2. Focus on the current section and subsections; write as though the "
        "analysis has already been completed.\n"
        "3. Produce 1–3 compact, cohesive paragraphs per subsection key.\n"
        "4. Use formal academic tone; avoid tutorial-style or meta-commentary.\n"
        "5. Avoid redundancy and restating objectives or what the section is "
        "supposed to do.\n"
        "6. Apply the dynamic forbidden block internally only (do not echo):\n"
        "{dynamic_forbidden_block}\n"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert academic writing assistant."),
            ("human", template),
        ]
    )


def build_assignment_content_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[Dict[str, Any], Dict[str, str]]:
    """
    Build a chain that generates section + subsection content as JSON.

    Input dict keys:
      - section_number: str
      - section_title: str
      - target_pages: float
      - target_words_total: int
      - courseplan_analysis: str
      - user_responses: str
      - graph_context_block: str
      - web_search_block: str
      - excel_context_snippet: str
      - citation_instruction: str
      - dynamic_forbidden_block: str
      - subsections: list[SubsectionSpec]

    Output:
      - Dict[str, str] mapping "main" and subsection keys to text.
    """

    prompt = _build_prompt()

    def to_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        import json

        subsections: List[SubsectionSpec] = inputs.get("subsections", [])
        return {
            "section_number": inputs.get("section_number", ""),
            "section_title": inputs.get("section_title", ""),
            "target_pages": inputs.get("target_pages", 1.0),
            "target_words_total": inputs.get("target_words_total", 800),
            "courseplan_analysis": inputs.get("courseplan_analysis", ""),
            "user_responses": inputs.get("user_responses", ""),
            "graph_context_block": inputs.get("graph_context_block", ""),
            "web_search_block": inputs.get("web_search_block", ""),
            "excel_context_snippet": inputs.get("excel_context_snippet", ""),
            "citation_instruction": inputs.get("citation_instruction", ""),
            "dynamic_forbidden_block": inputs.get("dynamic_forbidden_block", ""),
            "subsections_json": json.dumps(
                [
                    {
                        "key": sub["key"],
                        "title": sub["title"],
                        "requested_pages": sub["pages"],
                    }
                    for sub in subsections
                ],
                ensure_ascii=False,
            ),
        }

    llm_chain = prompt | llm | RunnableLambda(lambda msg: msg.content)  # type: ignore[arg-type]

    def escape_latex_in_json(json_str: str) -> str:
        """
        Escape unescaped backslashes in JSON string values to fix LaTeX commands.
        Preserves already-escaped sequences like \\n, \\t, \\\\.

        Strategy: In JSON strings, escape backslashes that aren't part of
        complete valid escape sequences. For unicode escapes, we need 4 hex digits.
        For others, it's a single character, but we need to be careful with
        LaTeX commands that start with valid escape chars (e.g., \frac vs \f).
        """
        result = []
        i = 0
        in_string = False

        while i < len(json_str):
            char = json_str[i]

            # Check if we're entering/exiting a string
            if char == '"':
                # Check if this quote is escaped
                # Count consecutive backslashes before this quote
                backslash_count = 0
                j = i - 1
                while j >= 0 and json_str[j] == "\\":
                    backslash_count += 1
                    j -= 1
                # If even number of backslashes (or zero), quote is not escaped
                if backslash_count % 2 == 0:
                    in_string = not in_string
                result.append(char)
                i += 1
                continue

            if in_string and char == "\\":
                # We're in a string and found a backslash
                # Strategy: Only preserve complete, valid JSON escape sequences
                # Everything else (LaTeX commands) gets escaped
                if i + 1 < len(json_str):
                    next_char = json_str[i + 1]

                    # Handle \uXXXX (unicode escape) - needs exactly 4 hex digits
                    if next_char == "u":
                        if i + 5 < len(json_str):
                            hex_chars = json_str[i + 2 : i + 6]
                            hex_digits = "0123456789abcdefABCDEF"
                            if all(c in hex_digits for c in hex_chars):
                                # Valid unicode escape - keep as is
                                result.append(char)
                                result.append(next_char)
                                result.append(json_str[i + 2])
                                result.append(json_str[i + 3])
                                result.append(json_str[i + 4])
                                result.append(json_str[i + 5])
                                i += 6
                                continue
                        # Not a complete unicode escape - escape it (LaTeX)
                        result.append("\\")
                        result.append("\\")
                        result.append(next_char)
                        i += 2
                        continue

                    # Always valid JSON escapes: ", \, /
                    if next_char in {'"', "\\", "/"}:
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue

                    # Control characters: \n, \r, \t are always valid
                    if next_char in {"n", "r", "t"}:
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue

                    # \b and \f: valid only if NOT followed by a letter (otherwise LaTeX like \begin, \frac)
                    if next_char in {"b", "f"}:
                        if i + 2 < len(json_str) and json_str[i + 2].isalpha():
                            # Followed by letter - this is LaTeX (e.g., \begin, \frac)
                            result.append("\\")
                            result.append("\\")
                            result.append(next_char)
                            i += 2
                            continue
                        # Not followed by letter - valid escape
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue

                    # Everything else is LaTeX - escape it
                    result.append("\\")
                    result.append("\\")
                    result.append(next_char)
                    i += 2
                    continue
                else:
                    # Backslash at end of string - escape it
                    result.append("\\")
                    result.append("\\")
                    i += 1
                    continue

            # Regular character
            result.append(char)
            i += 1

        return "".join(result)

    def parse_json_output(raw: str) -> Dict[str, str]:
        import json
        import re
        import logging

        logger = logging.getLogger(__name__)

        # Log raw output for debugging
        logger.debug(f"Raw LLM output length: {len(raw)}")
        logger.debug(f"Raw LLM output (first 500 chars): {raw[:500]}")

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
        # This avoids matching things like {1} or {citation} as JSON
        json_match = re.search(r'\{\s*"[^"]+"\s*:', cleaned, re.DOTALL)
        if json_match:
            # Found start of JSON object, now find the matching closing brace
            # Start from the opening brace
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
            else:
                # No matching brace found, use the original cleaned string
                logger.warning("JSON extraction: No matching closing brace found")
                pass

        # Log extracted JSON length to detect truncation
        logger.debug(f"Extracted JSON length: {len(cleaned)}")

        # Apply LaTeX escaping BEFORE attempting to parse
        # This ensures all LaTeX backslashes are properly escaped from the start
        escaped_json = escape_latex_in_json(cleaned)
        logger.debug("Applied LaTeX escaping to JSON before parsing")

        # Try to parse the escaped JSON
        data = None
        try:
            data = json.loads(escaped_json)
            logger.info(f"Successfully parsed JSON with keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed after LaTeX escaping: {e}")
            logger.error(f"Attempted to parse (first 500 chars): {escaped_json[:500]}")
            # Fall through to regex extraction

        if data is None:
            # Parsing failed - try regex extraction as fallback
            logger.warning("JSON parsing failed, trying regex extraction as fallback")
            return _extract_json_with_regex(cleaned, logger)

        # Ensure all values are strings and clean them
        result = {}
        empty_keys = []
        critical_keys = ["main"]  # Keys that are critical and should not be empty
        
        for k, v in data.items():
            # Convert value to string and clean up any JSON artifacts
            str_value = str(v)
            # Check if value is empty or just whitespace
            if not str_value or not str_value.strip():
                empty_keys.append(k)
                is_critical = k in critical_keys
                logger.warning(
                    f"JSON key '{k}' has empty or whitespace-only value"
                    + (" (CRITICAL - this section will be incomplete)" if is_critical else "")
                )
                # For critical keys, try to use a placeholder to avoid complete failure
                if is_critical:
                    logger.warning(f"Using placeholder content for critical key '{k}'")
                    str_value = f"[Content for {k} section was not generated. This section needs to be completed manually.]"
                else:
                    # Skip non-critical empty keys
                    continue
            # Remove any remaining JSON structure if it somehow got into the content
            if str_value.strip().startswith("{") and '"main"' in str_value:
                # This looks like JSON structure in the content, try to extract actual text
                str_value = re.sub(r'\{[^}]*"main"\s*:\s*"([^"]+)"', r"\1", str_value)
                str_value = re.sub(r'"[^"]+"\s*:\s*"([^"]+)"', r" \1", str_value)
                str_value = re.sub(r'[{}"]', "", str_value)
            # Log if equations are present (for debugging)
            if "$" in str_value or "\\" in str_value:
                logger.debug(f"Key '{k}' contains equations (has $ or \\)")
            result[str(k)] = str_value
            logger.debug(f"Added key '{k}' with content length: {len(str_value)}")

        # Log summary of empty values
        if empty_keys:
            logger.warning(
                f"Found {len(empty_keys)} empty value(s): {empty_keys}. "
                f"Result has {len(result)} valid key(s) out of {len(data)} total."
            )
        
        # If result is completely empty or missing critical sections, log error
        if not result:
            logger.error("JSON parsing resulted in empty dictionary!")
            logger.error(
                f"Original data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
            )
            logger.error(f"Original data: {str(data)[:500]}")
        elif any(k in empty_keys and k in critical_keys for k in critical_keys):
            logger.error(
                f"Critical section(s) have empty values: {[k for k in critical_keys if k in empty_keys]}. "
                "The generated content may be incomplete."
            )

        return result

    def _extract_json_with_regex(json_str: str, logger) -> Dict[str, str]:
        """
        Fallback: Extract key-value pairs from malformed JSON using regex.
        This is a last resort when standard JSON parsing fails.
        """
        import re

        result = {}

        # Pattern to match "key": "value" pairs
        # Handles multi-line values and escaped quotes
        pattern = r'"([^"]+)":\s*"((?:[^"\\]|\\.)*)"'

        matches = re.finditer(pattern, json_str, re.DOTALL)
        for match in matches:
            key = match.group(1)
            value = match.group(2)
            # Unescape the value
            # Unescape JSON escape sequences while preserving LaTeX backslashes
            # JSON escapes: \", \\, \/, \b, \f, \n, \r, \t
            # We need to preserve LaTeX commands like \begin, \frac, \sqrt, etc.
            # Strategy: unescape JSON control sequences but keep LaTeX backslashes
            # First, handle control characters
            value = value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            # For backslashes: \\\\ (double backslash in JSON) should become \\ (for LaTeX line breaks)
            # But \\ (escaped backslash in JSON) should become \ (for LaTeX commands)
            # The regex already handled escaped quotes, so we just need to handle backslashes
            # Use a safe placeholder (not NULL bytes)
            placeholder = "__DOUBLE_BACKSLASH_PLACEHOLDER__"
            value = value.replace(
                "\\\\\\\\", placeholder
            )  # Preserve intentional doubles
            value = value.replace("\\\\", "\\")  # Replace double with single
            value = value.replace(placeholder, "\\\\")  # Restore intentional doubles

            if value.strip():
                result[key] = value
                logger.info(f"Extracted key '{key}' via regex (length: {len(value)})")
                if "$" in value or "\\" in value:
                    logger.debug(f"  Key '{key}' contains equations")

        if result:
            logger.info(
                f"Regex extraction found {len(result)} keys: {list(result.keys())}"
            )
        else:
            logger.error("Regex extraction found no keys")
            # Last resort: try to extract just the main content
            main_match = re.search(
                r'"main"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL
            )
            if main_match:
                main_value = main_match.group(1)
                # Unescape JSON escapes while preserving LaTeX
                main_value = (
                    main_value.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "\r")
                )
                # Handle backslashes with safe placeholder
                placeholder = "__DOUBLE_BACKSLASH_PLACEHOLDER__"
                main_value = main_value.replace("\\\\\\\\", placeholder)
                main_value = main_value.replace("\\\\", "\\")
                main_value = main_value.replace(placeholder, "\\\\")
                result["main"] = main_value
                logger.warning("Extracted only 'main' key as last resort")
            else:
                # Ultimate fallback
                result["main"] = json_str
                logger.error("Could not extract any JSON, using raw string as 'main'")

        return result

    return (
        RunnableLambda(to_model_inputs) | llm_chain | RunnableLambda(parse_json_output)
    )
