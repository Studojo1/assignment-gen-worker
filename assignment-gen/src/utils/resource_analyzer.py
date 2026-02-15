"""
Resource Analyzer for Dynamic Question Generation.

This module analyzes uploaded files and web search results to provide
context-aware summaries for the assignment input collector.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from models.assignment import AssignmentSpec

logger = logging.getLogger(__name__)


class ResourceAnalyzer:
    """Analyzes resources to provide context for question generation."""

    @staticmethod
    def analyze_resources(spec: AssignmentSpec) -> str:
        """
        Analyze all available resources in the assignment spec and return a summary string.
        
        This summary is intended to be included in the LLM prompt to help it generate
        more relevant and contextual questions.
        """
        parts = []
        
        # 1. Analyze Uploaded Files
        if spec.uploaded_files:
            file_summaries = []
            for i, file_info in enumerate(spec.uploaded_files, 1):
                filename = file_info.get("filename", f"File {i}")
                extracted_text = file_info.get("extracted_text", "")
                
                if not extracted_text:
                    continue
                    
                # Basic content analysis
                # In a real implementation, this could use an LLM or more advanced NLP
                # For now, we'll extract key metadata and a preview
                
                content_len = len(extracted_text)
                preview = extracted_text[:500].replace("\n", " ")
                
                # Simple keyword detection for context
                keywords = []
                lower_text = extracted_text.lower()
                
                if "syllabus" in lower_text or "course plan" in lower_text:
                    keywords.append("Course Plan/Syllabus")
                if "assignment" in lower_text or "homework" in lower_text or "project" in lower_text:
                    keywords.append("Assignment Description")
                if "rubric" in lower_text or "grading" in lower_text:
                    keywords.append("Grading Rubric")
                if "chapter" in lower_text or "reading" in lower_text:
                    keywords.append("Reading Material")
                
                file_type_hint = ", ".join(keywords) if keywords else "General Document"
                
                file_summaries.append(
                    f"File {i}: {filename} ({file_type_hint})\n"
                    f"   - Length: {content_len} chars\n"
                    f"   - Preview: {preview}..."
                )
            
            if file_summaries:
                parts.append("=== UPLOADED RESOURCES ANALYSIS ===")
                parts.append(
                    "The user has uploaded the following files. Use this context to ask specific questions "
                    "about the assignment requirements found in these documents:"
                )
                parts.extend(file_summaries)
                parts.append("")

        # 2. Analyze Web Search Results
        if spec.web_search_results:
            web_summaries = []
            for i, result in enumerate(spec.web_search_results[:3], 1):  # Analyze top 3
                title = result.get("title", "Unknown")
                snippet = result.get("snippet", "")
                web_summaries.append(f"Source {i}: {title} - {snippet}")
            
            if web_summaries:
                parts.append("=== WEB SEARCH CONTEXT ===")
                parts.append(
                    "Relevant web sources found. You can ask the user if they want to focus on specific aspects "
                    "mentioned in these sources:"
                )
                parts.extend(web_summaries)
                parts.append("")

        # 3. Synthesize Recommendations
        if parts:
            parts.append("=== QUESTION GENERATION STRATEGY ===")
            parts.append(
                "1. PRIORITIZE questions based on the uploaded files (e.g., 'I see the rubric mentions X, should I focus on that?')."
            )
            parts.append(
                "2. If web sources are present, ask about their relevance or specific angles to explore."
            )
            parts.append(
                "3. AVOID asking for information that is clearly present in the file previews above."
            )
            
        return "\n".join(parts)
