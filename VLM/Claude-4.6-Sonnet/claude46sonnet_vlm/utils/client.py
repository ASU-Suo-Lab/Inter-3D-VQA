from __future__ import annotations

import base64
from io import BytesIO
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

from claude46sonnet_vlm.config.common import DEFAULT_MODEL, FORWARD_DEFAULTS, PROVIDER_NAME
from claude46sonnet_vlm.utils.io import ensure
from claude46sonnet_vlm.utils.secrets import resolve_api_key


def _resize_image_bytes(path: Path, *, max_edge: int) -> tuple[str, str]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        longest_edge = max(width, height)
        output_format = "JPEG"
        mime_type = "image/jpeg"
        if longest_edge > max_edge:
            scale = max_edge / float(longest_edge)
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format=output_format, quality=90, optimize=True)
        return mime_type, base64.b64encode(buffer.getvalue()).decode("ascii")


class Claude46SonnetClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = int(FORWARD_DEFAULTS["max_output_tokens"]),
        max_image_edge: int = int(FORWARD_DEFAULTS["max_image_edge"]),
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
        self.max_image_edge = int(max_image_edge)
        self.api_key = resolve_api_key()

    def answer(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> str:
        content = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            mime_type, encoded = _resize_image_bytes(image_path, max_edge=self.max_image_edge)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded,
                    },
                }
            )
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        request = urllib.request.Request(
            url="https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._perform_request(request)
        text = self._extract_text(body)
        ensure(text.strip(), f"{PROVIDER_NAME} returned an empty answer.")
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
        content = payload.get("content")
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
