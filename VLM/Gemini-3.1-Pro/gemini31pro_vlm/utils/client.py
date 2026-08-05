from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from gemini31pro_vlm.config.common import DEFAULT_MODEL, FORWARD_DEFAULTS, PROVIDER_NAME
from gemini31pro_vlm.utils.io import ensure
from gemini31pro_vlm.utils.secrets import resolve_api_key


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "image/jpeg"


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class Gemini31ProClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = int(FORWARD_DEFAULTS["max_output_tokens"]),
        timeout_seconds: int = 180,
        max_retries: int = int(FORWARD_DEFAULTS["max_retries"]),
        retry_backoff_seconds: float = float(FORWARD_DEFAULTS["retry_backoff_seconds"]),
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.api_key = resolve_api_key()

    def _token_budgets(self) -> list[int]:
        budgets = [self.max_output_tokens]
        for candidate in (1024, 2048):
            if self.max_output_tokens < candidate and candidate not in budgets:
                budgets.append(candidate)
        return budgets

    def _supports_zero_thinking_budget(self) -> bool:
        model = self.model.strip().lower()
        return "gemini-2.5-flash" in model

    def answer(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> str:
        parts = [{"text": user_prompt}]
        for image_path in image_paths:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": _guess_mime_type(image_path),
                        "data": _encode_image(image_path),
                    }
                }
            )
        token_budgets = self._token_budgets()
        last_empty_error: RuntimeError | None = None
        for max_output_tokens in token_budgets:
            payload = self._build_payload(
                system_prompt=system_prompt,
                parts=parts,
                max_output_tokens=max_output_tokens,
            )
            request = urllib.request.Request(
                url=self._build_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            body = self._perform_request(request)
            text = self._extract_text(body)
            if text.strip():
                return text.strip()
            last_empty_error = RuntimeError(self._build_empty_response_error(body, max_output_tokens))
            if not self._should_retry_empty_response(body, max_output_tokens, token_budgets[-1]):
                raise last_empty_error
        assert last_empty_error is not None
        raise last_empty_error

    def _build_payload(self, *, system_prompt: str, parts: list[dict[str, object]], max_output_tokens: int) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "temperature": self.temperature,
            "maxOutputTokens": int(max_output_tokens),
            "mediaResolution": FORWARD_DEFAULTS["media_resolution"],
        }
        if self._supports_zero_thinking_budget():
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    def _build_url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model, safe='')}:generateContent"
            f"?key={urllib.parse.quote(self.api_key, safe='')}"
        )

    def _perform_request(self, request: urllib.request.Request) -> dict[str, object]:
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"{PROVIDER_NAME} API request failed with status {exc.code}: {message}")
                if attempt <= self.max_retries and exc.code in {408, 429, 500, 502, 503, 504}:
                    time.sleep(self._retry_delay_seconds(exc, attempt))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"{PROVIDER_NAME} API request failed: {exc}")
                if attempt <= self.max_retries:
                    time.sleep(self._retry_delay_seconds(None, attempt))
                    continue
                raise last_error from exc
        assert last_error is not None
        raise last_error

    def _retry_delay_seconds(self, exc: urllib.error.HTTPError | None, attempt: int) -> float:
        if exc is not None:
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self.retry_backoff_seconds * (2 ** (attempt - 1))

    @staticmethod
    def _extract_text(payload: dict[str, object]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return ""
        chunks: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _candidate_finish_reasons(payload: dict[str, object]) -> list[str]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return []
        reasons: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            finish_reason = candidate.get("finishReason")
            if isinstance(finish_reason, str) and finish_reason:
                reasons.append(finish_reason)
        return reasons

    @staticmethod
    def _prompt_feedback_reason(payload: dict[str, object]) -> str | None:
        feedback = payload.get("promptFeedback")
        if not isinstance(feedback, dict):
            return None
        for key in ("blockReason", "block_reason"):
            value = feedback.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _should_retry_empty_response(self, payload: dict[str, object], max_output_tokens: int, max_retry_tokens: int) -> bool:
        finish_reasons = set(self._candidate_finish_reasons(payload))
        if "MAX_TOKENS" in finish_reasons and max_output_tokens < max_retry_tokens:
            return True
        return False

    def _build_empty_response_error(self, payload: dict[str, object], max_output_tokens: int) -> str:
        finish_reasons = self._candidate_finish_reasons(payload)
        prompt_feedback_reason = self._prompt_feedback_reason(payload)
        details: list[str] = []
        if finish_reasons:
            details.append(f"finish_reasons={finish_reasons}")
        if prompt_feedback_reason:
            details.append(f"prompt_feedback={prompt_feedback_reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if "MAX_TOKENS" in finish_reasons:
            return (
                f"{PROVIDER_NAME} returned no visible answer because the response exhausted "
                f"maxOutputTokens={max_output_tokens}{suffix}."
            )
        if prompt_feedback_reason is not None:
            return f"{PROVIDER_NAME} returned no visible answer because the prompt was blocked{suffix}."
        if finish_reasons:
            return f"{PROVIDER_NAME} returned no visible answer{suffix}."
        return f"{PROVIDER_NAME} returned an empty answer."
