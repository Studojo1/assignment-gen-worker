"""
Project: Studojo Agents
File: src/tools/graph.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 17 November 2025
Description: Langchain Tools related to Graph Generation using Matplotlib
"""

import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.graph_data_validator import validate_graph_data
from tools.common.web import search_and_fetch

logger = logging.getLogger(__name__)


class GraphGenerationResult(BaseModel):
    """Result of graph generation."""

    image_path: str = Field(description="Path to the generated image file")
    success: bool = Field(description="Whether the graph was generated successfully")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class GraphSpecification(BaseModel):
    """Specification for a graph to be generated from content."""

    title: str = Field(description="Title/description of the graph")
    graph_type: str = Field(description="Type of graph (bar, line, pie, scatter, etc.)")
    data_points: Dict[str, Any] = Field(
        description="Key data points or values to visualize, extracted from content"
    )
    x_axis_label: Optional[str] = Field(None, description="Label for x-axis")
    y_axis_label: Optional[str] = Field(None, description="Label for y-axis")
    rationale: str = Field(description="Why this graph is relevant to the content")
    validation_issues: Optional[List[str]] = Field(
        default=[], description="List of validation issues found"
    )
    confidence_score: float = Field(
        default=0.0, description="Confidence score of data accuracy"
    )


async def _extract_numerical_data(
    content: str, llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract numerical data from content text using LLM (replacing regex-based extraction).

    Args:
        content: Text content to analyze
        llm: Optional LLM instance for extraction. If None, falls back to simple regex.

    Returns:
        Dictionary with extracted numerical data points
    """
    # Try LLM-based extraction if available
    if llm:
        try:
            from chains import build_shared_llm
            from chains.utils import build_numerical_data_extraction_chain

            shared_llm = build_shared_llm(llm)
            extraction_chain = build_numerical_data_extraction_chain(shared_llm)
            result = await extraction_chain.ainvoke(content)

            # Convert to expected format
            extracted_data = {
                "percentages": result.get("percentages", []),
                "statistics": result.get("statistics", []),
                "years": result.get("years", []),
                "values": result.get("values", []),
                "comparisons": result.get("comparisons", []),
            }
            logger.info(
                f"Extracted numerical data using LLM: {len(extracted_data['percentages'])} percentages, {len(extracted_data['values'])} values"
            )
            return extracted_data
        except Exception as e:
            logger.warning(
                f"LLM-based numerical extraction failed: {e}, falling back to regex"
            )

    # Fallback to regex-based extraction
    extracted_data = {
        "percentages": [],
        "statistics": [],
        "years": [],
        "values": [],
        "comparisons": [],
    }

    # Extract percentages (e.g., "25%", "50 percent")
    percentage_pattern = r"(\d+(?:\.\d+)?)\s*%"
    percentages = re.findall(percentage_pattern, content)
    extracted_data["percentages"] = [float(p) for p in percentages]

    # Extract years (e.g., "2020", "2021-2023")
    year_pattern = r"\b(19|20)\d{2}\b"
    years = re.findall(year_pattern, content)
    extracted_data["years"] = list(set(years))

    # Extract statistics-like phrases (e.g., "increased by 50%", "decreased from X to Y")
    stat_patterns = [
        r"(?:increased|decreased|rose|fell|grew|declined)\s+(?:by|to|from)\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:times|fold)",
        r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
    ]
    for pattern in stat_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            extracted_data["statistics"].extend(
                matches
                if isinstance(matches[0], str)
                else [item for sublist in matches for item in sublist]
            )

    # Extract standalone numbers that might be values
    # This is a simple heuristic - look for numbers with context words
    value_context = r"(?:revenue|sales|profit|cost|price|amount|total|value|count|number|size)\s+(?:of|is|was|are|were)\s+(\d+(?:\.\d+)?)"
    values = re.findall(value_context, content, re.IGNORECASE)
    extracted_data["values"] = [float(v) for v in values]

    return extracted_data


async def analyze_content_for_graphs(
    content: str,
    max_graphs: int = 2,
    web_sources: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
) -> List[GraphSpecification]:
    """
    Analyze assignment content to identify relevant graphs using LLM.

    Args:
        content: The assignment content text to analyze
        max_graphs: Maximum number of graphs to suggest (default: 2)
        web_sources: Optional web sources for validation
        llm: Optional LLM instance for numerical data extraction

    Returns:
        List of GraphSpecification objects with graph details
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Extract numerical data first using LLM if available
    numerical_data = await _extract_numerical_data(content, llm=llm)

    # CHECK: If we have very few data points, try to search for more data
    # We consider it "sparse" if we have fewer than 3 values or years
    is_sparse_data = (
        len(numerical_data.get("values", [])) < 3
        and len(numerical_data.get("years", [])) < 3
        and len(numerical_data.get("percentages", [])) < 3
    )

    # If explicit web sources weren't provided or we need more data, try to search
    augmented_web_sources = list(web_sources) if web_sources else []

    if is_sparse_data and llm:
        logger.info(
            "Data appears sparse in content, attempting to search for statistical data..."
        )
        try:
            # Generate a specific query for data/statistics
            # We want to find tables, stats, and figures
            search_prompt = f"""
            Based on this assignment content, generate a search query specifically to find STATISTICAL DATA, 
            CHARTS, or TRENDS that would be relevant for a graph.
            
            Content: {content[:1000]}...
            
            Return ONLY the search query (e.g., "global electric vehicle sales 2020-2025 statistics").
            """

            # Simple direct query generation if helper fails or for speed
            from langchain_core.messages import HumanMessage, SystemMessage

            query = ""
            if hasattr(llm, "invoke"):
                response = llm.invoke(
                    [
                        SystemMessage(
                            content="You are a research assistant. Return ONLY a search query."
                        ),
                        HumanMessage(content=search_prompt),
                    ]
                )
                query = response.content.strip().replace('"', "").replace("'", "")

            if query:
                logger.info(f"Searching for data with query: {query}")
                search_results = await search_and_fetch(query, max_results=3)
                if search_results:
                    augmented_web_sources.extend(search_results)
                    logger.info(
                        f"Added {len(search_results)} web sources with potential data"
                    )
        except Exception as e:
            logger.warning(f"Failed to search for additional data: {e}")

    # Truncate content if too long (keep first 15000 chars for context)
    content_preview = content[:15000] + "..." if len(content) > 15000 else content

    # Prepare web context string
    web_context_str = ""
    if augmented_web_sources:
        web_context_str = "\n\nSupplementary Data from Web Search:\n"
        for i, source in enumerate(augmented_web_sources):
            # Limit each source content to avoid token overflow
            src_content = source.get("content", "") or source.get("snippet", "")
            web_context_str += f"Source {i+1} ({source.get('title', 'Unknown')}): {src_content[:2000]}...\n\n"

    prompt = f"""Analyze the following assignment content and identified web data to identify {max_graphs} relevant graphs that would enhance understanding.

Content:
{content_preview}
{web_context_str}

Extracted numerical data points (from original content):
{numerical_data}

Your task:
1. Identify what graphs would be most relevant and meaningful based on the content AND supplementary web data.
2. Extract or infer data points from either the content or web data.
3. For each graph, provide:
   - A descriptive title
   - Graph type (bar, line, pie, scatter, area, etc.)
   - Data points (as a dictionary with labels and values)
   - Axis labels
   - Rationale for why this graph enhances the content

IMPORTANT:
- You CAN use data found in the "Supplementary Data from Web Search" section.
- If data is qualitative in the main content but specific numbers appear in the web data, USE THE WEB DATA NUMBERS.
- Construct a COMPLETE dataset for the graph (at least 3-5 data points if possible).
- Real data is better than estimated, but if you see a clear trend described (e.g. "doubled every year"), you can infer the mathematical values.
- Return results as a JSON array of objects


Format your response as a JSON array with this structure:
[
  {{
    "title": "Descriptive graph title",
    "graph_type": "bar|line|pie|scatter|area",
    "data_points": {{"label1": value1, "label2": value2, ...}},
    "x_axis_label": "X-axis label",
    "y_axis_label": "Y-axis label",
    "rationale": "Why this graph is relevant"
  }}
]

Return ONLY the JSON array, no markdown, no explanations."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Remove markdown code fences if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        import json

        graph_specs_data = json.loads(response_text)

        # Convert to GraphSpecification objects and validate
        graph_specs = []
        for spec_data in graph_specs_data[:max_graphs]:
            spec = GraphSpecification(**spec_data)

            # Validate data points
            # Pass the augmented web sources to validation
            is_valid, confidence, issues = validate_graph_data(
                spec, content, augmented_web_sources
            )
            spec.confidence_score = confidence
            spec.validation_issues = issues

            if is_valid:
                graph_specs.append(spec)
            else:
                logger.warning(f"Graph '{spec.title}' failed validation: {issues}")

        logger.info(
            f"Analyzed content and found {len(graph_specs)} relevant and validated graphs"
        )
        return graph_specs

    except Exception as e:
        logger.error(f"Error analyzing content for graphs: {e}")
        # Return empty list on error rather than crashing
        return []


def _generate_matplotlib_code(
    description: str,
    data: Optional[Dict[str, Any]] = None,
    content_context: Optional[str] = None,
) -> str:
    """
    Use Anthropic API to generate matplotlib Python code for creating a graph.

    Args:
        description: Description of what graph to create
        data: Optional data dictionary with keys/values to plot
        content_context: Optional content text to provide context for the graph

    Returns:
        Python code string that generates the graph
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Build prompt for code generation
    data_context = ""
    if data:
        data_context = (
            f"\n\nAvailable data:\n{data}\n\nUse this data to create the graph."
        )

    content_context_str = ""
    if content_context:
        # Limit content context to first 2000 chars to avoid token limits
        content_preview = (
            content_context[:2000] + "..."
            if len(content_context) > 2000
            else content_context
        )
        content_context_str = (
            f"\n\nContext from assignment content:\n{content_preview}\n\n"
            "Ensure the graph visualizes concepts/data that are relevant to this content context."
        )

    prompt = f"""Generate Python code using matplotlib to create a graph based on this description:

{description}{data_context}{content_context_str}

Requirements:
- Use matplotlib.pyplot (import as plt)
- IMPORTANT: The variable 'output_path' is already defined and contains the full path where you MUST save the file
- You MUST use this exact variable name: output_path
- Save the figure using: plt.savefig(output_path, dpi=300, bbox_inches='tight')
- Do NOT hardcode any file path - always use the 'output_path' variable
- Make the graph professional and publication-ready
- Include proper labels, titles, and legends
- Use appropriate colors and styling
- Close the figure with plt.close() after saving

Matplotlib API Guidelines:
- When setting tick labels, use ax.set_xticks() or ax.set_yticks() BEFORE calling set_xticklabels() or set_yticklabels()
- Example: ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['A', 'B', 'C'])
- Avoid calling set_ticklabels() without first calling set_ticks() with the same number of ticks
- Use plt.tight_layout() before saving if needed for better spacing

CRITICAL: Do NOT include any code that calls external APIs (including Claude API, Anthropic API, or any other API). Only generate matplotlib plotting code. Do not import or use anthropic, openai, or any other API libraries.

Example code structure:
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Your plotting code here
# ... (create your graph)

# Save to the provided output_path variable
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

Return ONLY the Python code, no markdown code fences, no explanations. The code should be ready to execute."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        code = message.content[0].text.strip()

        # Remove markdown code fences if present
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        return code
    except Exception as e:
        logger.error(f"Error generating matplotlib code with Anthropic API: {e}")
        raise


def _execute_matplotlib_code(code: str, output_path: str) -> Tuple[bool, Optional[str]]:
    """
    Execute matplotlib code in a safe environment.

    Args:
        code: Python code string to execute
        output_path: Path where the image should be saved

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Set matplotlib backend to non-interactive 'Agg' before importing pyplot
        # This is critical for headless environments
        import matplotlib

        matplotlib.use("Agg")

        # Create a safe execution environment
        exec_globals = {
            "__builtins__": __builtins__,
            "plt": None,  # Will be imported
            "np": None,  # Will be imported
            "output_path": output_path,
        }

        # Import matplotlib and numpy after setting backend
        import matplotlib.pyplot as plt
        import numpy as np

        exec_globals["plt"] = plt
        exec_globals["np"] = np

        # Execute the code
        exec(code, exec_globals)

        # Verify that the file was actually created
        if not os.path.exists(output_path):
            error_msg = f"Code executed but file not found at {output_path}"
            logger.warning(error_msg)
            return False, error_msg

        return True, None
    except Exception as e:
        error_msg = f"Error executing matplotlib code: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Code that failed:\n{code}")
        import traceback

        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False, error_msg


@tool
def generate_matplotlib_graph(
    description: str,
    data: Optional[Dict[str, Any]] = None,
    output_filename: Optional[str] = None,
) -> GraphGenerationResult:
    """
    Generate a matplotlib graph based on a description using Anthropic API to create the code.

    Args:
        description: Description of what graph to create (e.g., "Create a bar chart showing
            sales by quarter", "Plot a line graph of temperature over time")
        data: Optional dictionary with data to plot. Keys can be used as variable names in the code.
        output_filename: Optional filename for the output image. If not provided, a temporary file
            will be created.

    Returns:
        GraphGenerationResult with the path to the generated image file
    """
    try:
        # Generate Python code using Anthropic API
        code = _generate_matplotlib_code(description, data)

        # Determine output path
        if output_filename:
            # Ensure it's in a temp directory or specified location
            if not os.path.isabs(output_filename):
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, output_filename)
            else:
                output_path = output_filename
        else:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            fd, output_path = tempfile.mkstemp(
                suffix=".png", prefix="graph_", dir=temp_dir
            )
            os.close(fd)  # Close the file descriptor, we'll use the path

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Execute the code to generate the graph
        success, exec_error = _execute_matplotlib_code(code, output_path)

        # Check if file exists at the expected path immediately after execution
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                logger.info(
                    f"Successfully generated graph at {output_path} (size: {file_size} bytes)"
                )
                return GraphGenerationResult(
                    image_path=output_path, success=True, error=None
                )
            else:
                logger.warning(f"Graph file created but is empty: {output_path}")

        # If not found at expected path, check current working directory
        # (sometimes generated code saves to cwd instead of output_path)
        cwd = os.getcwd()
        filename_only = os.path.basename(output_path)
        cwd_path = os.path.join(cwd, filename_only)

        if os.path.exists(cwd_path):
            file_size = os.path.getsize(cwd_path)
            if file_size > 0:
                logger.info(
                    f"Graph found in current directory: {cwd_path} (size: {file_size} bytes), using it instead of {output_path}"
                )
                return GraphGenerationResult(
                    image_path=cwd_path, success=True, error=None
                )

        # Also check for common alternative filenames that might have been generated
        # Check if any .png files were recently created in cwd or temp directory
        try:
            import glob
            import time

            current_time = time.time()
            search_dirs = [cwd]
            if output_dir and output_dir != cwd:
                search_dirs.append(output_dir)

            for search_dir in search_dirs:
                png_files = glob.glob(os.path.join(search_dir, "*.png"))
                # Check files modified in last 60 seconds
                for png_file in png_files:
                    try:
                        if os.path.getmtime(png_file) > current_time - 60:
                            file_size = os.path.getsize(png_file)
                            if file_size > 0:
                                logger.info(
                                    f"Found recently created graph file: {png_file} (size: {file_size} bytes), using it"
                                )
                                return GraphGenerationResult(
                                    image_path=png_file, success=True, error=None
                                )
                    except OSError:
                        continue
        except Exception as e:
            logger.debug(f"Error checking for recent PNG files: {e}")

        # If we get here, file wasn't found
        error_msg = (
            exec_error
            or "Failed to generate graph image - file not found at expected location"
        )
        logger.error(f"Graph generation failed: {error_msg}")
        logger.error(f"Expected output path: {output_path}")
        logger.error(f"Current working directory: {cwd}")
        return GraphGenerationResult(
            image_path="",
            success=False,
            error=error_msg,
        )

    except Exception as e:
        logger.error(f"Error in generate_matplotlib_graph: {e}")
        return GraphGenerationResult(image_path="", success=False, error=str(e))


# Global variable to store LLM instance - will be set by agent
_agent_llm_for_graph: Optional[Any] = None


def set_agent_llm_for_graph(llm: Any) -> None:
    """Set the LLM instance for graph tools."""
    global _agent_llm_for_graph
    _agent_llm_for_graph = llm


@tool
async def generate_content_aware_graph(
    content: str,
    section_title: Optional[str] = None,
    output_filename: Optional[str] = None,
    web_sources: Optional[List[Dict[str, Any]]] = None,
) -> GraphGenerationResult:
    """
    Generate a graph that is relevant to the provided content using LLM analysis.

    Args:
        content: The assignment content text to analyze and generate graph for
        section_title: Optional section title for context
        output_filename: Optional filename for the output image

    Returns:
        GraphGenerationResult with the path to the generated image file
    """
    try:
        # First, analyze content to find relevant graphs
        # Use global LLM if available
        llm_to_use = _agent_llm_for_graph
        graph_specs = await analyze_content_for_graphs(
            content, max_graphs=1, web_sources=web_sources, llm=llm_to_use
        )

        if not graph_specs:
            return GraphGenerationResult(
                image_path="",
                success=False,
                error="No relevant graphs found in content",
            )

        # Use the first (most relevant) graph specification
        spec = graph_specs[0]

        # Build description with data points
        description = f"{spec.title} ({spec.graph_type} chart)"
        if spec.x_axis_label and spec.y_axis_label:
            description += f" showing {spec.y_axis_label} vs {spec.x_axis_label}"

        # Prepare data for graph generation
        graph_data = {
            "graph_type": spec.graph_type,
            "data_points": spec.data_points,
            "x_axis_label": spec.x_axis_label or "X-axis",
            "y_axis_label": spec.y_axis_label or "Y-axis",
            "title": spec.title,
        }

        # Generate code with content context
        code = _generate_matplotlib_code(
            description=description, data=graph_data, content_context=content
        )

        # Determine output path
        if output_filename:
            if not os.path.isabs(output_filename):
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, output_filename)
            else:
                output_path = output_filename
        else:
            temp_dir = tempfile.gettempdir()
            fd, output_path = tempfile.mkstemp(
                suffix=".png", prefix="graph_", dir=temp_dir
            )
            os.close(fd)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Execute the code to generate the graph
        success, exec_error = _execute_matplotlib_code(code, output_path)

        # Check if file exists
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                logger.info(
                    f"Successfully generated content-aware graph at {output_path} (size: {file_size} bytes)"
                )
                return GraphGenerationResult(
                    image_path=output_path, success=True, error=None
                )

        error_msg = exec_error or "Failed to generate graph image"
        logger.error(f"Content-aware graph generation failed: {error_msg}")
        return GraphGenerationResult(
            image_path="",
            success=False,
            error=error_msg,
        )

    except Exception as e:
        logger.error(f"Error in generate_content_aware_graph: {e}")
        return GraphGenerationResult(image_path="", success=False, error=str(e))


# Image embedding functionality is in tools.doc.add_image_to_document
