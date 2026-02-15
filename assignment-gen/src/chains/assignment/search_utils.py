"""
Utility helpers for building safe, concise web-search queries from assignment prompts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

async def generate_search_query_with_llm(
    llm: BaseChatModel, 
    description: str, 
    course_name: Optional[str] = None,
    uploaded_files_context: Optional[str] = None
) -> str:
    """
    Use an LLM to extract a high-quality search query from the assignment description and uploaded files.
    
    Args:
        llm: The language model to use
        description: Assignment description/prompt
        course_name: Optional course name
        uploaded_files_context: Optional context from uploaded files (extracted text/preview)
    """
    if not description and not uploaded_files_context:
        return ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a search query generator for academic assignments. Your task is to extract a single, "
                    "concise, and highly relevant web search query that will find authoritative, citable sources.\n\n"
                    "Critical Rules:\n"
                    "1. Extract the CORE TOPIC from BOTH the assignment description AND uploaded file content.\n"
                    "2. If uploaded files are provided, they contain the actual assignment requirements - prioritize "
                    "key terms, topics, and requirements from those files over generic description text.\n"
                    "3. Ignore noise like 'Contact Support', 'I need help', 'Please write', etc.\n"
                    "4. Focus on the CORE ACADEMIC TOPIC and KEY TERMS. Extract the main subject matter.\n"
                    "5. Prioritize finding authoritative sources - include terms like 'guide', 'manual', 'documentation', "
                    "'best practices', 'official', 'academic', or domain-specific authoritative terms when appropriate.\n"
                    "6. For procedural/task-oriented topics (e.g., 'how to write work instructions'), search for: "
                    "'how to write work instructions guide' or 'work instructions best practices' (not just 'work instructions').\n"
                    "7. For company/financial topics, prioritize official sources: 'Tech Mahindra annual report' or "
                    "'Tech Mahindra financial statements official'.\n"
                    "8. Include the course name or domain context ONLY if it adds specificity.\n"
                    "9. Use 3-8 words maximum. Be specific but concise.\n"
                    "10. Avoid generic terms that return low-quality results. Use precise terminology from the assignment files/description.\n"
                    "11. The query MUST be directly relevant to what the assignment is asking for. Do NOT search for generic or unrelated topics.\n"
                    "12. If the content is purely noise or you cannot determine a relevant topic, return 'NO_QUERY'.\n"
                    "13. Return ONLY the query string, no quotes, no explanations, no punctuation at the end.\n\n"
                    "Examples:\n"
                    "- Description mentions 'work instructions' + file talks about manufacturing → 'how to write work instructions guide'\n"
                    "- Description mentions 'Tech Mahindra' + file has financial analysis → 'Tech Mahindra annual report financial statements'\n"
                    "- File content about quantum gates → 'quantum computing gates introduction'\n"
                    "- Description about Roman history → 'Roman Empire history'\n"
                ),
            ),
            (
                "human",
                (
                    "{assignment_context}"
                    "\n\nGenerate a precise search query that will find authoritative, citable sources "
                    "RELEVANT TO THE ASSIGNMENT TOPIC:\n"
                ),
            ),
        ]
    )

    # Build context string from description and uploaded files
    context_parts = []
    
    if uploaded_files_context and uploaded_files_context.strip():
        context_parts.append("=== UPLOADED FILES (PRIORITY - Contains actual assignment requirements) ===\n")
        context_parts.append(uploaded_files_context.strip())
        context_parts.append("\n")
    
    if description and description.strip():
        context_parts.append("=== ASSIGNMENT DESCRIPTION ===\n")
        context_parts.append(description.strip())
        context_parts.append("\n")
    
    if course_name and course_name.strip():
        context_parts.append(f"Course: {course_name.strip()}\n")
    
    assignment_context = "\n".join(context_parts).strip()
    
    if not assignment_context:
        return ""

    chain = prompt | llm | RunnableLambda(lambda x: x.content.strip())
    
    try:
        query = await chain.ainvoke({"assignment_context": assignment_context})
        
        if query == "NO_QUERY" or not query:
            return ""
        
        # Clean up the query: remove quotes, extra whitespace, trailing punctuation
        query = query.strip()
        # Remove surrounding quotes if LLM added them
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            query = query[1:-1].strip()
        
        # Remove trailing punctuation (periods, commas) that might interfere with search
        query = query.rstrip('.,;:!?')
        
        # Normalize whitespace
        query = " ".join(query.split())
        
        # Validate: query should have at least 2 words and be reasonable length
        words = query.split()
        if len(words) < 1:
            return ""
        if len(query) > 100:  # Too long, likely malformed
            # Take first 8 words max
            query = " ".join(words[:8])
        
        return query.strip()
    except Exception as e:
        logger.warning(f"Failed to generate search query with LLM: {e}")
        return ""


NOISE_KEYWORDS = {
    "attachment",
    "attached",
    "see attachment",
    "uploaded file",
    "refer attachment",
    "courseplan",
    "course plan",
    "course-plan",
    "pdf",
    "docx",
    "ppt",
}

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "assignment",
    "at",
    "be",
    "by",
    "for",
    "from",
    "generate",
    "i",
    "in",
    "is",
    "make",
    "of",
    "on",
    "please",
    "report",
    "study",
    "support",
    "contact",
    "task",
    "that",
    "the",
    "this",
    "to",
    "with",
    "write",
}


@dataclass
class SearchTopicResult:
    """Structured information about the sanitized search topic."""

    query: str
    selected_sentence: str
    removed_lines: List[str]


def _clean_description(description: str) -> tuple[str, List[str]]:
    removed: List[str] = []
    clean_lines: List[str] = []

    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if "http://" in lower or "https://" in lower:
            removed.append(line)
            continue

        if any(keyword in lower for keyword in NOISE_KEYWORDS):
            removed.append(line)
            continue

        clean_lines.append(line)

    return " ".join(clean_lines), removed


def _select_sentence(text: str) -> str:
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text

    keyword_sentences = [
        sentence
        for sentence in sentences
        if any(token in sentence.lower() for token in ["vision", "mission", "analysis", "strategy"])
        or "assignment" in sentence.lower()
        or "report" in sentence.lower()
        or "essay" in sentence.lower()
    ]

    if keyword_sentences:
        return keyword_sentences[0].strip()

    # Fallback: find the first sentence that has enough content (e.g. > 3 words)
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 3:
            return sentence.strip()

    return sentences[0].strip()


def _trim_to_keywords(sentence: str, max_words: int) -> str:
    tokens = re.findall(r"\b[\w'-]+\b", sentence)
    keywords: List[str] = []

    for token in tokens:
        lower = token.lower()
        if lower in STOP_WORDS:
            continue
        if len(token) <= 2:
            continue
        keywords.append(token)
        if len(keywords) >= max_words:
            break

    if not keywords:
        return ""

    return " ".join(keywords[:max_words])


def extract_search_topic(description: Optional[str], max_words: int = 12) -> SearchTopicResult:
    """
    Extract a concise topic suitable for web search queries from the assignment description.
    """
    if not description:
        return SearchTopicResult(query="", selected_sentence="", removed_lines=[])

    cleaned_text, removed_lines = _clean_description(description)
    sentence = _select_sentence(cleaned_text)
    query = _trim_to_keywords(sentence, max_words) if sentence else ""
    return SearchTopicResult(query=query.strip(), selected_sentence=sentence, removed_lines=removed_lines)


def build_legacy_search_query(description: str, course_name: Optional[str] = None, max_words: int = 12) -> str:
    """
    Legacy heuristics for constructing a search query. Used as a fallback when sanitization fails.
    """
    import re as _re

    search_parts: List[str] = []
    description_lower = description.lower()

    topic_match = _re.search(
        r"(?:(?:assignment\s+)?(?:report|essay|paper|paperwork)|assignment)\s+(?:on|about|regarding|for)\s+([^,.!?\n]+?)(?:\.|,|!|\?|$|\s+This|\s+for|\s+is|\s+I\s)",
        description_lower,
        _re.IGNORECASE,
    )
    if topic_match:
        topic = topic_match.group(1).strip()
        topic = _re.sub(r"\s+(?:this|for|is|i|a|an|the)\s*$", "", topic, flags=_re.IGNORECASE).strip()
        topic = _re.sub(r"^\b(?:a|an|the|on|about|regarding|for)\s+", "", topic, flags=_re.IGNORECASE).strip()
        if topic and len(topic.split()) <= max_words:
            search_parts.append(topic)

    if not search_parts:
        topic_match = _re.search(
            r"(?:on|about|regarding|for)\s+([A-Z][^,.!?\n]+?)(?:\.|,|!|\?|$|This|for|is|I)",
            description,
            _re.IGNORECASE,
        )
        if topic_match:
            topic = topic_match.group(1).strip()
            topic = _re.sub(r"\b(?:a|an|the|on|about|regarding|for)\b", "", topic, flags=_re.IGNORECASE).strip()
            if topic and len(topic.split()) <= max_words:
                search_parts.append(topic)

    if not search_parts:
        capitalized_phrases = _re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", description)
        meaningful = [p for p in capitalized_phrases if len(p.split()) <= 5]
        if meaningful:
            search_parts.append(meaningful[0])

    if not search_parts:
        words = description.split()[:15]
        stop_words = {
            "i",
            "need",
            "to",
            "make",
            "an",
            "assignment",
            "report",
            "on",
            "about",
            "regarding",
            "for",
            "a",
            "the",
            "this",
            "is",
            "for",
            "course",
            "called",
            "support",
            "contact",
        }
        key_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if key_words:
            search_parts.append(" ".join(key_words[:max_words]))

    if course_name and len(course_name.split()) <= 5:
        aggregate = " ".join(search_parts).lower()
        if course_name.lower() not in aggregate:
            search_parts.append(course_name)

    query = " ".join(search_parts).strip()
    if not query:
        return ""

    query_words = query.split()
    return " ".join(query_words[:max_words])


