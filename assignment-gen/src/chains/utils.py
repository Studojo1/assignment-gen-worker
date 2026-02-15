"""
Project: Studojo Assignment Generator
File: src/chains/utils.py
Description: Utility chains for assignment generation (numerical data extraction, file summarization).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

logger = logging.getLogger(__name__)


def build_numerical_data_extraction_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[str, Dict[str, Any]]:
    """
    Extract numerical data from content using LLM (replacing regex-based extraction).

    Input: content text (string)
    Output: dict with:
      - percentages: List[float]
      - statistics: List[Dict[str, Any]] - statistical phrases with values
      - years: List[str]
      - values: List[Dict[str, Any]] - key-value pairs (e.g., revenue, sales)
      - comparisons: List[Dict[str, Any]] - comparative data (from X to Y)
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a data extraction assistant. Extract numerical data from text content.\n\n"
                    "Respond with STRICT JSON, no explanation, using this shape:\n"
                    '{{"percentages": [float], "statistics": [{{"phrase": "...", "value": float, "type": "increase/decrease/etc"}}], '
                    '"years": [str], "values": [{{"label": "...", "value": float, "unit": "..."}}], '
                    '"comparisons": [{{"from": float, "to": float, "change": float, "type": "..."}}]}}\n\n'
                    "Extract:\n"
                    "1. Percentages: any percentage values (e.g., '25%', '50 percent')\n"
                    "2. Statistics: phrases like 'increased by 50%', 'decreased from X to Y', 'rose 3-fold'\n"
                    "3. Years: any year mentions (e.g., '2020', '2021-2023')\n"
                    "4. Values: key metrics with labels (e.g., 'revenue is $100M', 'sales of 500 units')\n"
                    "5. Comparisons: before/after values, changes over time\n\n"
                    "Be thorough but accurate. Only extract data that is clearly numerical and contextual.\n"
                ),
            ),
            ("human", "{content}"),
        ]
    )

    chain = prompt | llm | RunnableLambda(lambda msg: msg.content)

    def parse_json(output: str) -> Dict[str, Any]:
        try:
            cleaned = output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            # Ensure all fields exist and are lists
            result = {
                "percentages": parsed.get("percentages", []),
                "statistics": parsed.get("statistics", []),
                "years": parsed.get("years", []),
                "values": parsed.get("values", []),
                "comparisons": parsed.get("comparisons", []),
            }
            # Validate types
            if not isinstance(result["percentages"], list):
                result["percentages"] = []
            if not isinstance(result["statistics"], list):
                result["statistics"] = []
            if not isinstance(result["years"], list):
                result["years"] = []
            if not isinstance(result["values"], list):
                result["values"] = []
            if not isinstance(result["comparisons"], list):
                result["comparisons"] = []
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"Failed to parse numerical data extraction JSON: {e}, output: {output[:200]}"
            )
            return {
                "percentages": [],
                "statistics": [],
                "years": [],
                "values": [],
                "comparisons": [],
            }

    return chain | RunnableLambda(parse_json)


def build_file_summarization_chain(
    llm: BaseChatModel,
) -> RunnableSerializable[Dict[str, Any], str]:
    """
    Generate AI summary of file content.

    Input: dict with:
      - filename: str
      - file_type: str
      - content: str (full or preview)
      - max_length: Optional[int] - max summary length
    Output: summary string
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a file summarization assistant. Generate a concise, informative summary of file content.\n\n"
                    "Guidelines:\n"
                    "1. Provide a clear overview of the file's main topics and purpose\n"
                    "2. Highlight key points, findings, or important information\n"
                    "3. If it's an academic document, note the subject area and main arguments\n"
                    "4. If it's a data file, summarize the data structure and key metrics\n"
                    "5. Keep the summary concise but informative\n"
                    "6. If the content is truncated, note that in the summary\n"
                    "7. Focus on actionable insights and main themes\n\n"
                    "Return only the summary text, no additional commentary."
                ),
            ),
            (
                "human",
                (
                    "File: {filename}\n"
                    "Type: {file_type}\n"
                    "Content length: {content_length} characters\n\n"
                    "Content:\n{content}\n\n"
                    "Generate a summary:"
                ),
            ),
        ]
    )

    def to_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        content = inputs.get("content", "")
        # Truncate content if too long for summarization
        if len(content) > 10000:
            content = content[:10000] + "\n\n[Content truncated for summarization...]"
        return {
            "filename": inputs.get("filename", "Unknown"),
            "file_type": inputs.get("file_type", "unknown"),
            "content": content,
            "content_length": len(inputs.get("content", "")),
        }

    chain = (
        RunnableLambda(to_model_inputs)
        | prompt
        | llm
        | RunnableLambda(lambda msg: msg.content.strip())
    )

    return chain
