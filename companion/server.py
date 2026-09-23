import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class ServerError(RuntimeError):
    pass


class LlamaServer:
    """Owns the llama-server.exe process: start it, wait until the model is loaded, stop it."""

    def __init__(self, binary: Path, model: Path, host: str, port: int,
                 context_tokens: int, gpu_layers: int, log_dir: Path, lora: Path | None = None,
                 profile=None):
        self.lora = lora
        self.profile = profile  # a ModelProfile adds its own args (vision, KV cache, thinking)
        self.binary = binary
        self.model = model
        self.host = host
        self.port = port
        self.context_tokens = context_tokens
        self.gpu_layers = gpu_layers
        self.log_path = log_dir / "llama-server.log"
        self._proc: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def health(self) -> str:
        """'ok' when ready, 'loading' while the model loads, 'down' if nothing answers."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                return json.loads(resp.read()).get("status", "ok")
        except urllib.error.HTTPError as e:
            # llama-server answers 503 while the model is still loading.
            return "loading" if e.code == 503 else "down"
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return "down"

    def start(self, timeout: float = 180) -> bool:
        """Start the server, or reuse one already on this port. Returns True if we launched it."""
        if self.health() == "ok":
            return False
        for path in (self.binary, self.model):
            if not path.exists():
                raise ServerError(f"Missing file: {path}")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")
        args = self.args()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(args, stdout=self._log_file, stderr=subprocess.STDOUT,
                                      creationflags=flags)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerError(f"llama-server exited early (code {self._proc.returncode}).\n"
                                  f"{self._log_tail()}")
            if self.health() == "ok":
                return True
            time.sleep(0.5)
        self.stop()
        raise ServerError(f"llama-server not ready after {timeout:.0f}s.\n{self._log_tail()}")

    def args(self) -> list[str]:
        args = [str(self.binary), "-m", str(self.model), "--host", self.host, "--port", str(self.port),
                "-ngl", str(self.gpu_layers)]
        if self.profile is not None:
            args += self.profile.server_args()
        else:
            args += ["-c", str(self.context_tokens)]
        if self.lora:
            args += ["--lora", str(self.lora)]
        return args

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _log_tail(self, lines: int = 15) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log)"
        return "\n".join(text.splitlines()[-lines:])
