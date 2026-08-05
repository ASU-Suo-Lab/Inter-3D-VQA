from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

from gpt54_vlm.config.common import DEFAULT_MODEL, FORWARD_DEFAULTS, PROVIDER_NAME
from gpt54_vlm.utils.secrets import resolve_api_key


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "image/jpeg"


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


class GPT54Client:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = int(FORWARD_DEFAULTS["max_output_tokens"]),
        timeout_seconds: int = 180,
        max_retries: int = int(FORWARD_DEFAULTS["max_retries"]),
        retry_backoff_seconds: float = float(FORWARD_DEFAULTS["retry_backoff_seconds"]),
    ) -> None:
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.api_key = resolve_api_key()

    def answer(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> str:
        content = [{"type": "input_text", "text": user_prompt}]
        for image_path in image_paths:
            mime_type = _guess_mime_type(image_path)
            image_b64 = _encode_image(image_path)
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_b64}",
                    "detail": FORWARD_DEFAULTS["image_detail"],
                }
            )
        payload = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": FORWARD_DEFAULTS["reasoning_effort"]},
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": content},
            ],
        }
        request = urllib.request.Request(
            url="https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._perform_request(request)
        text = self._extract_text(body)
        if not text.strip():
            raise RuntimeError(self._build_empty_response_error(body, self.max_output_tokens))
        return text.strip()

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
                if self._should_retry_http_error(exc.code, message, attempt):
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

    def _should_retry_http_error(self, status_code: int, message: str, attempt: int) -> bool:
        if attempt > self.max_retries:
            return False
        if status_code == 429 and self._looks_like_quota_error(message):
            return False
        return status_code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _looks_like_quota_error(message: str) -> bool:
        lowered = message.lower()
        return "insufficient_quota" in lowered or "exceeded your current quota" in lowered

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
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output = payload.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    text = piece.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)
        return ""

    @staticmethod
    def _build_empty_response_error(payload: dict[str, object], max_output_tokens: int) -> str:
        status = payload.get("status")
        incomplete_details = payload.get("incomplete_details")
        if isinstance(incomplete_details, dict):
            reason = incomplete_details.get("reason")
            if status == "incomplete" and isinstance(reason, str):
                if reason == "max_output_tokens":
                    return (
                        f"{PROVIDER_NAME} returned no visible answer because the response was incomplete and exhausted "
                        f"max_output_tokens={max_output_tokens}. Increase --max-output-tokens or reduce reasoning work."
                    )
                return f"{PROVIDER_NAME} returned no visible answer because the response was incomplete: reason={reason}."
        if status == "incomplete":
            return f"{PROVIDER_NAME} returned no visible answer because the response was incomplete."
        if status == "failed":
            return f"{PROVIDER_NAME} returned no visible answer because the response status was failed."
        return f"{PROVIDER_NAME} returned an empty answer."
