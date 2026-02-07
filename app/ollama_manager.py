import os
import time
import subprocess
from typing import Optional, Dict, Any

import requests


class OllamaManager:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_cmd: str = "ollama",
        auto_start: bool = True,
        startup_timeout_sec: int = 20,
        model: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_cmd = ollama_cmd
        self.auto_start = auto_start
        self.startup_timeout_sec = startup_timeout_sec
        self.model = model
        self.default_options = default_options or {}
        self._proc: Optional[subprocess.Popen] = None

    # ---------- Health / Status ----------
    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def wait_until_up(self) -> None:
        deadline = time.time() + self.startup_timeout_sec
        while time.time() < deadline:
            if self.is_up():
                return
            time.sleep(0.5)
        raise RuntimeError("Ollama did not become ready within timeout.")

    # ---------- Start / Stop ----------
    def start_if_needed(self) -> None:
        if self.is_up():
            return
        if not self.auto_start:
            raise RuntimeError("Ollama is not running and auto_start is disabled.")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            [self.ollama_cmd, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.wait_until_up()

    # ---------- Model ----------
    def model_exists(self, model: str) -> bool:
        try:
            tags = requests.get(f"{self.ollama_url}/api/tags", timeout=5).json()
            existing = {m.get("name") for m in tags.get("models", []) if isinstance(m, dict)}
            return model in existing
        except Exception:
            # if tags fails, assume unknown
            return False

    def ensure_model(self, model: str) -> None:
        if self.model_exists(model):
            return

        # Try pull via HTTP (if supported by your Ollama build)
        try:
            requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600,
            ).raise_for_status()
        except Exception:
            # Fallback: try CLI pull
            try:
                subprocess.run([self.ollama_cmd, "pull", model], check=True)
            except Exception:
                # Let generation fail loudly later if pull is impossible
                pass

    # ---------- Generate ----------
    def generate(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> str:
        used_model = model or self.model
        if not used_model:
            raise ValueError("Model is not set. Provide model=... or set manager.model.")

        payload = {
            "model": used_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        merged_options = dict(self.default_options)
        if options:
            merged_options.update(options)
        if merged_options:
            payload["options"] = merged_options

        r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["response"]
    
        # ---------- Shutdown ----------
    def shutdown(self) -> None:
        """
        Gracefully shutdown Ollama ONLY if it was started by this manager.
        """
        if self._proc is None:
            # Ollama was not started by us
            return

        try:
            if self._proc.poll() is None:
                # Try graceful terminate
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self._proc.kill()
        finally:
            self._proc = None


def build_manager_from_env() -> OllamaManager:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_cmd = os.getenv("OLLAMA_CMD", "ollama")
    auto_start = os.getenv("AUTO_START_OLLAMA", "1") == "1"
    startup_timeout_sec = int(os.getenv("OLLAMA_STARTUP_TIMEOUT_SEC", "20"))
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    default_options = {
        "temperature": 0,
        "top_p": 1,
        "num_predict": 512,
        # "seed": 42,  # enable only if supported
    }

    return OllamaManager(
        ollama_url=ollama_url,
        ollama_cmd=ollama_cmd,
        auto_start=auto_start,
        startup_timeout_sec=startup_timeout_sec,
        model=model,
        default_options=default_options,
    )

