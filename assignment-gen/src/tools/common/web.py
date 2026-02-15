"""
Project: Studojo Agents
File: src/tools/web.py
Author: Rithul Kamesh
Email: hi@rithul.dev
Created: 16 November 2025
Description: Langchain Tools related to Web Searching/Scraping
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field

# --- CRITICAL: SSL Configuration MUST happen before other imports ---
_ssl_disabled = os.getenv("DDGS_DISABLE_SSL_VERIFY", "false").lower() in (
    "true",
    "1",
    "yes",
)

if _ssl_disabled:
    import ssl
    import urllib3

    # 1. Disable SSL at Python level
    ssl._create_default_https_context = ssl._create_unverified_context

    # 2. Disable urllib3 warnings
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # 3. Set environment variables for underlying libraries
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HTTPX_VERIFY"] = "0"

    # 4. Monkey-patch httpx to default verify=False
    try:
        import httpx

        # Patch Client
        _original_client_init = httpx.Client.__init__

        def _patched_client_init(self, *args, **kwargs):
            if "verify" not in kwargs:
                kwargs["verify"] = False
            elif _ssl_disabled:
                kwargs["verify"] = False
            return _original_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init

        # Patch AsyncClient
        _original_async_client_init = httpx.AsyncClient.__init__

        def _patched_async_client_init(self, *args, **kwargs):
            if "verify" not in kwargs:
                kwargs["verify"] = False
            elif _ssl_disabled:
                kwargs["verify"] = False
            return _original_async_client_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_async_client_init

        logger.info("Aggressively patched httpx to disable SSL verification")

    except ImportError:
        pass

# Now import DDGS after SSL is configured
from ddgs import DDGS

# Try to import playwright - make it optional for Alpine compatibility
try:
    from playwright.async_api import (
        async_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

    # Create dummy objects so the code doesn't crash
    class DummyPlaywright:
        def __init__(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async_playwright = lambda: DummyPlaywright()
    PlaywrightTimeoutError = Exception


class WebSearchResults(BaseModel):
    title: str = Field(description="Title of the Search Result")
    url: str = Field(description="URL of the Result")
    snippet: str = Field(description="Small write-up of the website thing")


def _search_duckduckgo_impl(
    keyword: str, max_results: int = 10, max_retries: int = 3
) -> List[WebSearchResults]:
    """
    Internal implementation of DuckDuckGo search (non-tool version) with retry logic.

    Args:
        keyword: The search query to look up on DuckDuckGo
        max_results: Maximum number of results to return (default: 10)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        A list of WebSearchResults objects containing titles, URLs, and snippets
    """
    last_exception = None

    # Check if SSL verification should be disabled (for development/proxy environments)
    disable_ssl_verify = os.getenv("DDGS_DISABLE_SSL_VERIFY", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if disable_ssl_verify:
        logger.warning(
            "SSL verification is disabled for DuckDuckGo search. "
            "This should only be used in development or behind corporate proxies."
        )

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"DuckDuckGo search attempt {attempt + 1}/{max_retries} for query: {keyword}"
            )

            # Try to use DDGS - SSL context should already be configured at module level
            # If SSL is disabled, the context is set to unverified before DDGS import
            with DDGS() as ddgs:
                results = list(ddgs.text(keyword, max_results=max_results))

            if not results:
                logger.info(
                    f"DuckDuckGo search returned no results for query: {keyword}"
                )
                return []

            search_results = []
            for result in results:
                title = result.get("title", "No title")
                url = result.get("href", "No URL")
                snippet = result.get("body", "No description")

                search_results.append(
                    WebSearchResults(title=title, url=url, snippet=snippet)
                )

            logger.info(
                f"DuckDuckGo search succeeded: found {len(search_results)} results for query: {keyword}"
            )
            return search_results

        except Exception as e:
            last_exception = e
            error_type = type(e).__name__
            error_msg = str(e).lower()

            # Check if it's an SSL certificate error
            is_ssl_error = (
                "ssl" in error_msg
                or "certificate" in error_msg
                or "certificate verify failed" in error_msg
                or "CERTIFICATE_VERIFY_FAILED" in str(e)
            )

            if is_ssl_error:
                if disable_ssl_verify:
                    # SSL is disabled but still failing - likely httpx issue
                    logger.error(
                        f"DuckDuckGo search failed due to SSL certificate error even though "
                        f"DDGS_DISABLE_SSL_VERIFY=true is set. This indicates httpx SSL configuration issue. "
                        f"Skipping DuckDuckGo search. Error: {error_type}: {e}"
                    )
                    # Don't retry - SSL errors won't resolve
                    break
                else:
                    logger.warning(
                        f"DuckDuckGo search failed due to SSL certificate error. "
                        f"Set DDGS_DISABLE_SSL_VERIFY=true to disable SSL verification (development only). "
                        f"Skipping DuckDuckGo search. Error: {error_type}: {e}"
                    )
                    # Don't retry SSL errors - they won't resolve without config change
                    break
            else:
                logger.warning(
                    f"DuckDuckGo search attempt {attempt + 1}/{max_retries} failed: {error_type}: {e}"
                )

            # Don't retry on the last attempt or if it's an SSL error
            if attempt < max_retries - 1 and not is_ssl_error:
                # Exponential backoff: wait 2s, 4s, 8s... (increased from 1s, 2s...)
                wait_time = 2 ** (attempt + 1)
                logger.debug(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                if not is_ssl_error:
                    # If it's just "No results found", it's not really an error, just empty
                    if "no results found" in error_msg:
                        logger.info(
                            f"DuckDuckGo search returned no results for query: {keyword}"
                        )
                        return []

                    logger.error(
                        f"All {max_retries} DuckDuckGo search attempts failed for query: {keyword}. "
                        f"Last error: {error_type}: {e}",
                        exc_info=True,
                    )

    # If all retries failed, return empty list
    return []


@tool
def search_duckduckgo(keyword: str, max_results: int = 10) -> List[WebSearchResults]:
    """
    Search DuckDuckGo for the given keyword and return the search results.

    This is a LangChain tool wrapper around the implementation function.

    Args:
        keyword: The search query to look up on DuckDuckGo
        max_results: Maximum number of results to return (default: 10)

    Returns:
        A list of WebSearchResults objects containing titles, URLs, and snippets
    """
    if not keyword or not keyword.strip():
        return []
    return _search_duckduckgo_impl(keyword, max_results)


async def fetch_webpage_content(
    url: str, timeout: int = 10, max_length: int = 3000
) -> str:
    """
    Fetch and extract main text content from a webpage using playwright with requests fallback.
    """
    import requests
    from bs4 import BeautifulSoup

    def _requests_fallback(url, timeout):
        try:
            logger.info(f"Using requests fallback for {url}")
            # SSL verify=False if globally disabled
            verify_ssl = not (
                os.getenv("DDGS_DISABLE_SSL_VERIFY", "false").lower()
                in ("true", "1", "yes")
            )

            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                verify=verify_ssl,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "noscript",
                    "iframe",
                ]
            ):
                script.decompose()

            text = soup.get_text(separator=" ")
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.error(f"Requests fallback failed for {url}: {e}")
            return ""

    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Invalid URL: {url}")
            return ""

        # Try Playwright if available
        if PLAYWRIGHT_AVAILABLE:
            try:
                async with async_playwright() as p:
                    # Attempt to launch - this triggers error if system deps missing
                    browser = await p.chromium.launch(headless=True)

                    try:
                        context = await browser.new_context(
                            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        )
                        page = await context.new_page()

                        # Navigate
                        await page.goto(
                            url, wait_until="domcontentloaded", timeout=timeout * 1000
                        )

                        # Wait a bit for basic hydration
                        await page.wait_for_timeout(1000)

                        html_content = await page.content()

                        soup = BeautifulSoup(html_content, "html.parser")
                        for script in soup(
                            [
                                "script",
                                "style",
                                "nav",
                                "header",
                                "footer",
                                "aside",
                                "noscript",
                                "iframe",
                            ]
                        ):
                            script.decompose()

                        text = soup.get_text(separator=" ")
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (
                            phrase.strip()
                            for line in lines
                            for phrase in line.split("  ")
                        )
                        text = " ".join(chunk for chunk in chunks if chunk)

                        if len(text) > max_length:
                            text = text[:max_length] + "..."
                        return text

                    finally:
                        await browser.close()
            except Exception as e:
                logger.warning(
                    f"Playwright execution failed for {url} (falling back to requests): {e}"
                )
                # Fall through to requests
                pass
        else:
            logger.info("Playwright not installed/available, skipping to requests")

        # Fallback to requests (if Playwright unavailable OR failed)
        text = await asyncio.to_thread(_requests_fallback, url, timeout)

        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text

    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
        return ""


async def search_and_fetch(keyword: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search DuckDuckGo and fetch actual content from result URLs (recursive).

    Args:
        keyword: The search query
        max_results: Maximum number of results to fetch content for (default: 5)

    Returns:
        List of dicts with keys: title, url, snippet, content
    """
    try:
        logger.info(
            f"Starting search_and_fetch for query: {keyword} (max_results: {max_results})"
        )

        if not keyword or not keyword.strip():
            logger.warning("Empty search query provided to search_and_fetch")
            return []

        # Get search results (use impl directly, not the tool)
        search_results = _search_duckduckgo_impl(keyword, max_results=max_results)

        if not search_results:
            logger.warning(f"No search results found for query: {keyword}")
            return []

        logger.info(
            f"Found {len(search_results)} search results, fetching content for top {max_results}"
        )

        # Fetch content for each result
        enhanced_results = []
        for i, result in enumerate(search_results[:max_results], 1):
            logger.debug(
                f"Fetching content [{i}/{min(len(search_results), max_results)}]: {result.url}"
            )
            try:
                content = await fetch_webpage_content(result.url)
                enhanced_results.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "content": content,
                    }
                )
                # Log if content fetch failed
                if not content:
                    logger.warning(
                        f"Failed to fetch content from {result.url} "
                        "(empty content returned, but keeping result with snippet)"
                    )
            except Exception as fetch_error:
                logger.warning(
                    f"Error fetching content from {result.url}: {fetch_error}. "
                    "Including result with snippet only."
                )
                # Still include the result with just snippet if content fetch fails
                enhanced_results.append(
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "content": "",  # Empty content but keep the result
                    }
                )

        logger.info(
            f"search_and_fetch completed: {len(enhanced_results)} results with content for query: {keyword}"
        )
        return enhanced_results

    except Exception as e:
        logger.error(
            f"Error in search_and_fetch for query '{keyword}': {e}", exc_info=True
        )
        return []


@tool
def read_webpage_content(url: str) -> str:
    """
    Read the content of a specific webpage URL.

    Use this tool when you have a URL from search results and need to read its full content.
    Do NOT use this for uploaded files (use read_file_content instead).

    Args:
        url: The URL of the webpage to read.

    Returns:
        The text content of the webpage.
    """
    try:
        # Run async fetch in sync context
        return asyncio.run(fetch_webpage_content(url))
    except Exception as e:
        logger.error(f"Error reading webpage {url}: {e}", exc_info=True)
        return f"Error reading webpage: {str(e)}"
