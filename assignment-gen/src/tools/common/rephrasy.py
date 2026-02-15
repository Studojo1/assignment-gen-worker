"""
Project: Studojo Agents
File: src/tools/rephrasy.py
Author: Rithul Kamesh
Created: 17 November 2025
Description: Service wrapper for the Rephrasy Humanizer API.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class HumanizerError(RuntimeError):
    """Raised when the Rephrasy API fails."""


class RephrasyConfig(BaseModel):
    api_key: Optional[str]
    base_url: str = "https://v1-humanizer.rephrasy.ai/api"
    log_path: Optional[str] = None


class RephrasyService:
    """Service for interacting with the Rephrasy Humanizer API."""

    def __init__(self, config: Optional[RephrasyConfig] = None) -> None:
        api_key = os.getenv("REPHRASY_API_KEY") or os.getenv("REPHRASY_API_TOKEN")
        log_path = os.getenv("REPHRASY_LOG_PATH")
        self.config = config or RephrasyConfig(api_key=api_key, log_path=log_path)
        if self.config.log_path is None:
            self.config.log_path = os.path.join("logs", "rephrasy_requests.jsonl")

        if not self.config.api_key:
            logger.warning(
                "Missing REPHRASY_API_KEY/REPHRASY_API_TOKEN environment variable. "
                "Humanization will be a no-op fallback."
            )

    def _write_jsonl_log(self, record: Dict[str, Any]) -> None:
        """Append structured request/response log entries to JSONL."""
        log_path = self.config.log_path
        if not log_path:
            return
        try:
            directory = os.path.dirname(log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                json.dump(record, log_file, ensure_ascii=False)
                log_file.write("\n")
        except Exception:
            logger.warning("Failed to write Rephrasy JSONL log", exc_info=True)

    def _log_request_response(
        self,
        request_id: str,
        payload: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        order_id: Optional[str] = None,
    ) -> None:
        extra: Dict[str, Any] = {
            "request_id": request_id,
            "payload": payload,
            "order_id": order_id,
        }
        timestamp = datetime.now(timezone.utc).isoformat()
        log_record: Dict[str, Any] = {
            "timestamp": timestamp,
            "request_id": request_id,
            "order_id": order_id,
            "payload": payload,
        }
        if response is not None:
            extra["response"] = response
            log_record["response"] = response
            logger.info("Rephrasy request succeeded", extra=extra)
        if error is not None:
            extra["error"] = error
            log_record["error"] = error
            logger.error("Rephrasy request failed", extra=extra)
        self._write_jsonl_log(log_record)

    def _is_valid_language(
        self, text: str, llm: Optional["BaseChatModel"] = None
    ) -> bool:
        """
        Check if the text is in the target language (English) using LLM.
        If no LLM is provided or text is very short, assumes valid.
        """
        if not text or len(text.strip()) < 10:
            return True

        if not llm:
            # Fallback to loose heuristic if no LLM: check for high ratio of CJK characters
            # or just assume True if we can't check
            cjk_count = 0
            for char in text:
                if "\u4e00" <= char <= "\u9fff":
                    cjk_count += 1
            if len(text) > 0 and (cjk_count / len(text)) > 0.05:
                logger.warning(
                    f"Detected potential foreign language (CJK ratio > 5%): {text[:50]}..."
                )
                return False
            return True

        try:
            # Simple prompt to check language
            prompt = (
                "You are a language detector. Determine if the following text is primarily in English. "
                "Ignore code blocks or proper nouns. "
                "Respond with exactly 'YES' if it is English, or 'NO' if it is another language (e.g. Spanish, Chinese, Japanese, etc.).\n\n"
                f"Text: {text[:1000]}\n\n"  # Check first 1000 chars
                "Response (YES/NO):"
            )
            response = llm.invoke(prompt)
            content = (
                response.content.strip().upper()
                if hasattr(response, "content")
                else str(response).strip().upper()
            )

            is_valid = "YES" in content
            if not is_valid:
                logger.warning(
                    f"LLM detected non-English content: {content} - Text sample: {text[:50]}..."
                )
            return is_valid
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Assuming valid.")
            return True

    def humanize_text(
        self,
        content: str,
        model: str = "Undetectable Model",
        words: bool = True,
        order_id: Optional[str] = None,
        timeout: int = 120,
        llm: Optional["BaseChatModel"] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Call the Rephrasy Humanizer API with provided content.

        Args:
            content: Text to humanize
            model: Model to use
            words: Whether to preserve words
            order_id: Optional tracking ID
            timeout: Request timeout
            llm: Optional LLM for language validation
            max_retries: Number of retries on invalid language output

        Returns parsed JSON response.
        """

        request_id = str(uuid.uuid4())

        if not content or not content.strip():
            raise HumanizerError("Content must be a non-empty string.")

        if not self.config.api_key:
            raise HumanizerError("Rephrasy API key not configured.")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "text": content,
            "model": model,
            "words": True,
        }

        last_error = None

        for attempt in range(max_retries):
            current_request_id = f"{request_id}_{attempt}"
            self._log_request_response(current_request_id, payload, order_id=order_id)

            try:
                resp = requests.post(
                    self.config.base_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                response_data = resp.json()

                # Validate output language
                output_text = response_data.get("output", "")
                if output_text and not self._is_valid_language(output_text, llm):
                    logger.warning(
                        f"Attempt {attempt+1}/{max_retries}: Rephrasy returned invalid language. Retrying with original text..."
                    )
                    self._log_request_response(
                        current_request_id,
                        payload,
                        response=response_data,
                        order_id=order_id,
                        error={
                            "type": "invalid_language",
                            "message": "Detected non-English output",
                        },
                    )
                    # Ensure we retry with original content (payload already has it, but be explicit)
                    payload["text"] = content
                    continue

                self._log_request_response(
                    current_request_id,
                    payload,
                    response=response_data,
                    order_id=order_id,
                )
                return response_data

            except requests.exceptions.HTTPError as e:  # type: ignore[assignment]
                status = getattr(e.response, "status_code", 502)
                response_text = getattr(e.response, "text", "Upstream error")
                try:
                    response_json = e.response.json() if e.response else None
                except Exception:
                    response_json = None

                error_details = {
                    "status_code": status,
                    "response_text": response_text,
                    "response_json": response_json,
                    "headers": dict(e.response.headers) if e.response else None,
                }
                self._log_request_response(
                    current_request_id, payload, error=error_details, order_id=order_id
                )
                logger.exception(
                    "Rephrasy API HTTP error: %s - %s", status, response_text
                )
                last_error = HumanizerError(f"Rephrasy API error: {response_text}")

            except requests.exceptions.RequestException as e:
                error_details = {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "request_details": "Failed to reach Rephrasy API",
                }
                self._log_request_response(
                    current_request_id, payload, error=error_details
                )
                logger.exception("Rephrasy API request failed: %s", e)
                last_error = HumanizerError("Failed to reach Rephrasy API")

        # If we exhausted retries, fallback to original text instead of raising error
        logger.warning(
            f"Rephrasy failed after {max_retries} retries. Falling back to original text."
        )
        if last_error:
            logger.warning(f"Last error was: {last_error}")
        else:
            logger.warning(
                f"Failed to get valid English response after {max_retries} retries"
            )
        # Return original text as fallback
        return {"output": content}

    def humanize_text_or_fallback(
        self,
        text: str,
        order_id: Optional[str] = None,
        llm: Optional["BaseChatModel"] = None,
    ) -> str:
        """
        Humanize text with fallback to original text on error.
        """

        try:
            result = self.humanize_text(text, order_id=order_id, llm=llm)
            output = result.get("output")
            if isinstance(output, str) and output.strip():
                return output
            return text
        except HumanizerError:
            logger.warning("Humanizer failed or disabled, using original text")
            return text
