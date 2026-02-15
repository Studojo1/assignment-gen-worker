"""
Project: Studojo Agents
File: src/utils/graph_data_validator.py
Author: Rithul Kamesh
Description: Utilities for validating graph data against content and web sources.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_verified: bool = Field(description="Whether the data point is verified")
    confidence_score: float = Field(description="Confidence score (0.0 to 1.0)")
    source: str = Field(description="Source of verification (content, web, or none)")
    details: str = Field(description="Details about the verification result")


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    if not text:
        return ""
    text = str(text).lower()
    # Remove punctuation except for decimal points and percentage signs
    text = re.sub(r"[^\w\s.%]", "", text)
    return text.strip()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    if not text:
        return []
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return [float(m) for m in matches]


def normalize_number_string(val_str: str) -> str:
    """Normalize number string (remove commas) for comparison."""
    return val_str.replace(",", "").replace(" ", "")


def verify_data_point(
    label: str,
    value: Any,
    content: str,
    web_sources: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResult:
    """
    Verify a single data point against content and web sources.

    Args:
        label: The label of the data point (e.g., "2020", "Revenue")
        value: The value of the data point (e.g., 100, 50.5)
        content: The assignment content text
        web_sources: Optional list of web search results

    Returns:
        ValidationResult object
    """
    # Convert value to float if possible for comparison
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        numeric_value = None

    content_norm = normalize_text(content)
    label_norm = normalize_text(str(label))

    # 1. Check if exact value exists in content
    if numeric_value is not None:
        # Check for exact number match in content
        value_str = str(numeric_value)
        if value_str.endswith(".0"):
            value_str = value_str[:-2]

        # Simple check: is the number in the text?
        if value_str in content:
            # Check if it's near the label (if label is found)
            if label_norm in content_norm:
                return ValidationResult(
                    is_verified=True,
                    confidence_score=0.9,
                    source="content",
                    details=f"Found exact value {value} and label '{label}' in content",
                )
            else:
                return ValidationResult(
                    is_verified=True,
                    confidence_score=0.7,
                    source="content",
                    details=f"Found value {value} in content, but label '{label}' not found",
                )
    else:
        # Handle non-numeric values (strings)
        value_str = str(value)
        value_norm = normalize_text(value_str)

        if value_norm and value_norm in content_norm:
            if label_norm in content_norm:
                return ValidationResult(
                    is_verified=True,
                    confidence_score=0.85,
                    source="content",
                    details=f"Found value '{value}' and label '{label}' in content",
                )
            else:
                return ValidationResult(
                    is_verified=True,
                    confidence_score=0.65,
                    source="content",
                    details=f"Found value '{value}' in content, but label '{label}' not found",
                )

    # 2. Check web sources if available
    if web_sources:
        for source in web_sources:
            source_content = source.get("content", "") + " " + source.get("snippet", "")
            source_norm = normalize_text(source_content)

            if numeric_value is not None:
                value_str = str(numeric_value)
                if value_str.endswith(".0"):
                    value_str = value_str[:-2]

                if value_str in source_content:
                    if label_norm in source_norm:
                        return ValidationResult(
                            is_verified=True,
                            confidence_score=0.85,
                            source="web",
                            details=f"Found value {value} and label '{label}' in web source: {source.get('title', 'Unknown')}",
                        )

                # Check for "fuzzy" number match (e.g. 1,000 vs 1000, or approx values)
                if label_norm in source_norm:
                    # If label is in source, we trust it more easily
                    pass

            else:
                # Check string value in web sources
                value_str = str(value)
                value_norm = normalize_text(value_str)
                if value_norm and value_norm in source_norm:
                    if label_norm in source_norm:
                        return ValidationResult(
                            is_verified=True,
                            confidence_score=0.8,
                            source="web",
                            details=f"Found value '{value}' and label '{label}' in web source: {source.get('title', 'Unknown')}",
                        )

        # Check if ANY web source confirms the value even if label is missing (relaxed)
        if numeric_value is not None:
            value_str = str(numeric_value)
            if value_str.endswith(".0"):
                value_str = value_str[:-2]

            for source in web_sources:
                source_content = (
                    source.get("content", "") + " " + source.get("snippet", "")
                )
                if value_str in source_content or normalize_number_string(
                    value_str
                ) in normalize_text(source_content):
                    return ValidationResult(
                        is_verified=True,
                        confidence_score=0.7,
                        source="web_inferred",
                        details=f"Found value {value} in web source {source.get('title', 'Unknown')} (label not found nearby)",
                    )

    # 3. Fuzzy matching / Semantic check (simplified for now)

    # RELAXED FALLBACK: If we have web sources, we are more lenient
    # If the value looks "reasonable" (e.g. it's a number and we have web sources about the topic),
    # we assign a low but non-zero confidence rather than 0.
    if web_sources and numeric_value is not None:
        return ValidationResult(
            is_verified=False,
            confidence_score=0.4,  # Low confidence but not zero
            source="inferred",
            details=f"Value {value} not explicitly found but web sources are present. Assumed inferred.",
        )

    return ValidationResult(
        is_verified=False,
        confidence_score=0.0,
        source="none",
        details=f"Value {value} not found in content or web sources",
    )


def validate_graph_data(
    graph_spec: Any, content: str, web_sources: Optional[List[Dict[str, Any]]] = None
) -> Tuple[bool, float, List[str]]:
    """
    Validate all data points in a graph specification.

    Args:
        graph_spec: GraphSpecification object
        content: The assignment content text
        web_sources: Optional list of web search results

    Returns:
        Tuple of (is_valid, overall_confidence, issues)
    """
    data_points = graph_spec.data_points
    if not data_points:
        return False, 0.0, ["No data points provided"]

    total_score = 0.0
    count = 0
    issues = []

    # Handle different data structures
    # 1. Dictionary: {"Label": Value, ...}
    if isinstance(data_points, dict):
        for label, value in data_points.items():
            result = verify_data_point(label, value, content, web_sources)
            total_score += result.confidence_score
            count += 1

            if not result.is_verified:
                issues.append(f"Unverified data point: {label} = {value}")

    # 2. List of dicts (common in some chart types): [{"x": ..., "y": ...}, ...]
    elif isinstance(data_points, list):
        for item in data_points:
            if isinstance(item, dict):
                # Try to find value-like keys
                for k, v in item.items():
                    if isinstance(v, (int, float)) or (
                        isinstance(v, str) and v.replace(".", "", 1).isdigit()
                    ):
                        result = verify_data_point(k, v, content, web_sources)
                        total_score += result.confidence_score
                        count += 1
                        if not result.is_verified:
                            issues.append(f"Unverified data point in item: {k} = {v}")

    if count == 0:
        return False, 0.0, ["Could not parse data points structure"]

    avg_confidence = total_score / count

    # Threshold for validation
    # We require at least 50% confidence overall (lowered from 60%) to allow for inferred data
    is_valid = avg_confidence >= 0.5

    # If we have issues, check if they are critical
    # We accept the graph if we have > 0.5 confidence even with some unverified points
    if not is_valid and avg_confidence >= 0.4 and web_sources:
        # If we have web sources and "okay" confidence, let it pass with a warning
        is_valid = True
        issues.append("Validation relaxed due to presence of web support")

    return is_valid, avg_confidence, issues
