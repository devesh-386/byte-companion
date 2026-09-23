import _socket
import http.client
import json
import urllib.parse
from typing import Callable, Protocol

from .cancel import Cancelled, CancelToken

# content is text, or (for vision models) a list of parts:
#   [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
Message = dict


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def chat(self, messages: list[Message], *, stop: list[str] | None = None,
             on_token: Callable[[str], None] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             cancel: CancelToken | None = None) -> str: ...


class LlamaClient:
    """Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint, streaming."""

    def __init__(self, base_url: str, temperature: float = 0.4, max_tokens: int = 700,
                 timeout: float = 300):
        u = urllib.parse.urlparse(base_url)
        self.host, self.port = u.hostname, u.port or 80
        self.path = "/v1/chat/completions"
        self.url = f"{base_url}{self.path}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_timings: dict = {}

    def chat(self, messages: list[Message], *, stop: list[str] | None = None,
             on_token: Callable[[str], None] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             cancel: CancelToken | None = None) -> str:
        body = {
            "messages": messages,
            "stream": True,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            body["stop"] = stop
        if cancel:
            cancel.check()

        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        unregister = lambda: None
        parts: list[str] = []
        try:
            try:
                conn.connect()
            except OSError as e:
                raise LLMError(f"Cannot reach llama-server at {self.url}: {e}")
            if cancel:
                # Reads block while the model is still reading the prompt, so there's no loop to check a flag in.
                # Shutting the socket down from the cancelling thread makes the blocked read return at once,
                # and llama-server notices the disconnect and stops generating.
                unregister = cancel.on_cancel(lambda: _shutdown(conn.sock))
            conn.request("POST", self.path, body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise LLMError(f"llama-server returned {resp.status}: {resp.read().decode(errors='replace')[:300]}")
            # Server-Sent Events: each chunk arrives as a line "data: {json}", ending with "data: [DONE]".
            for raw_line in resp:
                if cancel and cancel.cancelled:
                    break
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("timings"):  # llama-server adds speed numbers to the last chunk
                    self.last_timings = chunk["timings"]
                choices = chunk.get("choices") or []
                token = (choices[0].get("delta") or {}).get("content") if choices else None
                if token:
                    parts.append(token)
                    if on_token:
                        on_token(token)
        except (OSError, http.client.HTTPException) as e:
            if cancel and cancel.cancelled:
                raise Cancelled(cancel.reason) from None
            raise LLMError(f"Lost connection to llama-server: {e}") from e
        finally:
            unregister()
            conn.close()
        if cancel and cancel.cancelled:
            raise Cancelled(cancel.reason)
        return "".join(parts)


def _shutdown(sock) -> None:
    # Measured on Windows: shutdown() and socket.close() both leave a read blocked in another thread
    # (close() is deferred while the HTTP response holds a file handle). Only the C-level close unblocks it.
    if sock is not None:
        try:
            _socket.socket.close(sock)
        except OSError:
            pass
