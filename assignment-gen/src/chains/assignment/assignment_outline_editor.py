"""
Chat-based outline editor for assignment outlines.

Accepts current outline + user message, returns updated outline + confirmation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)


@dataclass
class OutlineEditorOutput:
    """Output from the outline editor chain."""

    outline: Dict[str, Any]
    assistant_message: str


def _parse_editor_response(raw: str, current_outline: Dict[str, Any]) -> OutlineEditorOutput:
    """Parse LLM response into updated outline and assistant message."""
    cleaned = raw.strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline > 0:
            cleaned = cleaned[first_newline:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    # Try to fix trailing commas
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # Try to parse as JSON with both keys
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Extract first complete JSON object
        brace_count = 0
        start_pos = cleaned.find("{")
        if start_pos >= 0:
            for i in range(start_pos, len(cleaned)):
                if cleaned[i] == "{":
                    brace_count += 1
                elif cleaned[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            data = json.loads(cleaned[start_pos : i + 1])
                            break
                        except json.JSONDecodeError:
                            pass
            else:
                data = {}
        else:
            data = {}

    outline = data.get("updated_outline") or data.get("outline")
    assistant_message = data.get("assistant_message") or data.get("message") or "Outline updated."

    if not outline or not isinstance(outline, dict) or "sections" not in outline:
        logger.warning("Outline editor response missing valid outline, returning current")
        return OutlineEditorOutput(outline=current_outline, assistant_message=assistant_message)

    return OutlineEditorOutput(outline=outline, assistant_message=assistant_message)


def build_assignment_outline_editor_chain(llm: BaseChatModel):
    """
    Build a chain that edits an assignment outline based on user chat.

    Input: dict with
      - outline: current outline (dict with "sections" key)
      - user_message: user's edit request (e.g. "Add a section about X", "Remove section 2")

    Output: OutlineEditorOutput with updated outline and assistant_message.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant that helps edit assignment outlines.\n\n"
                    "You will receive the current outline (JSON) and a user message describing "
                    "changes they want. Apply the requested edits and return:\n"
                    "1. The updated outline as JSON (same structure: {{\"sections\": [...]}})\n"
                    "2. A brief confirmation message (e.g. \"Done. I've added a section about X.\")\n\n"
                    "You MUST respond with STRICT JSON only, no markdown, no extra text. Use this shape:\n"
                    '{{"updated_outline": {{"sections": [...]}}, "assistant_message": "Your brief confirmation here"}}\n\n'
                    "Section structure: each section has \"id\", \"title\", \"target_words\", and optionally "
                    '"subsections" (array of objects with "id", "title", "target_words"). '
                    'You may add "notes" for graph/table hints.\n\n'
                    "Preserve the overall structure. Keep section IDs consistent (e.g. 1, 2, 3 or 1.1, 1.2). "
                    "When adding sections, use the next available ID. When changing word counts, "
                    "adjust other sections so the total stays roughly the same if possible."
                ),
            ),
            (
                "human",
                (
                    "Current outline:\n{outline_json}\n\n"
                    "User request: {user_message}\n\n"
                    "Return ONLY valid JSON with \"updated_outline\" and \"assistant_message\"."
                ),
            ),
        ]
    )

    def to_inputs(inputs: Dict[str, Any]) -> Dict[str, str]:
        outline = inputs.get("outline") or {"sections": []}
        return {
            "outline_json": json.dumps(outline, indent=2),
            "user_message": inputs.get("user_message", ""),
        }

    def parse_output(raw: str, inputs: Dict[str, Any]) -> OutlineEditorOutput:
        current = inputs.get("outline") or {"sections": []}
        return _parse_editor_response(raw, current)

    llm_chain = prompt | llm | RunnableLambda(lambda msg: msg.content)  # type: ignore[arg-type]

    async def _run(inputs: Dict[str, Any]) -> OutlineEditorOutput:
        prepared = to_inputs(inputs)
        raw = await llm_chain.ainvoke(prepared)
        return parse_output(raw, inputs)

    return RunnableLambda(_run)
